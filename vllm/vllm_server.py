#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vLLM Server для PDF Converter Pipeline v4.0
OpenAI-совместимый сервер с поддержкой Qwen моделей для обработки китайских технических документов
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import time
from pathlib import Path

# FastAPI и веб-сервер
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# vLLM импорты
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid

# Pydantic модели для OpenAI API
from pydantic import BaseModel, Field
from typing import Literal, Optional

# Утилиты
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import psutil

# =======================================================================================
# КОНФИГУРАЦИЯ И ЛОГИРОВАНИЕ
# =======================================================================================

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = structlog.get_logger()

# Конфигурация из переменных окружения
class VLLMConfig:
    def __init__(self):
        # Основные настройки
        self.host = os.getenv("VLLM_HOST", "0.0.0.0")
        self.port = int(os.getenv("VLLM_PORT", "8000"))
        self.api_key = os.getenv("VLLM_API_KEY", "vllm-api-key")
        
        # Модели
        self.text_model = os.getenv("VLLM_TEXT_MODEL", "Qwen/Qwen2.5-32B-Instruct")
        self.vision_model = os.getenv("VLLM_VISION_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")
        self.current_model = self.text_model  # Начинаем с текстовой модели
        
        # Параметры производительности
        self.tensor_parallel_size = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "2"))
        self.max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
        self.gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
        self.trust_remote_code = os.getenv("VLLM_TRUST_REMOTE_CODE", "true").lower() == "true"
        
        # Кэш и хранилище
        self.hf_home = os.getenv("HF_HOME", "/mnt/storage/models/huggingface")
        
        # Убеждаемся что директории существуют
        Path(self.hf_home).mkdir(parents=True, exist_ok=True)

config = VLLMConfig()

# =======================================================================================
# PROMETHEUS МЕТРИКИ
# =======================================================================================

# Счетчики
request_counter = Counter('vllm_requests_total', 'Total requests', ['model', 'endpoint'])
error_counter = Counter('vllm_errors_total', 'Total errors', ['model', 'error_type'])

# Гистограммы
request_duration = Histogram('vllm_request_duration_seconds', 'Request duration', ['model', 'endpoint'])
token_generation = Histogram('vllm_tokens_generated', 'Tokens generated per request', ['model'])

# Gauges
active_requests = Gauge('vllm_active_requests', 'Active requests')
gpu_memory_usage = Gauge('vllm_gpu_memory_usage_bytes', 'GPU memory usage')
model_loaded = Gauge('vllm_model_loaded', 'Model loaded status', ['model'])

# =======================================================================================
# OPENAI API МОДЕЛИ (Pydantic)
# =======================================================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 30
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.1
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = 30
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "pdf-converter-pipeline"

# =======================================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =======================================================================================

async def update_gpu_metrics():
    """Обновление GPU метрик"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_usage.set(meminfo.used)
    except Exception as e:
        logger.warning(f"Could not update GPU metrics: {e}")

def get_specialized_system_prompt(task_type: str, source_lang: str = "zh-CN", target_lang: str = "ru") -> str:
    """Получение специализированного системного промпта для разных задач"""
    
    if task_type == "document_conversion":
        return """You are an expert in converting Chinese technical documents to high-quality Markdown.

DOCUMENT TYPE: Chinese server hardware documentation with IPMI/BMC/Redfish commands.

CRITICAL REQUIREMENTS:
1. STRUCTURE PRESERVATION: Maintain exact heading hierarchy, lists, tables, code blocks
2. TECHNICAL ACCURACY: Keep ALL technical commands unchanged (IPMI, BMC, Redfish, API calls)
3. TABLE FORMATTING: Convert to proper Markdown tables, preserve data alignment
4. CODE PRESERVATION: Wrap technical commands in code blocks
5. CHINESE CONTENT: Keep Chinese descriptions in Chinese
6. OUTPUT: Clean Markdown only, no explanations"""

    elif task_type == "translation":
        lang_names = {
            "zh-CN": "Chinese", "ru": "Russian", "en": "English"
        }
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        return f"""You are a professional translator specializing in technical documentation.

TRANSLATION TASK: {source_name} → {target_name}

CRITICAL RULES:
1. PRESERVE: All Markdown formatting, tables, code blocks, technical commands
2. DO NOT TRANSLATE: IPMI commands, BMC commands, Redfish API calls, parameter names
3. TRANSLATE ONLY: Descriptions, explanations, user-facing text
4. MAINTAIN: Document structure, numbering, indentation
5. OUTPUT: Only translated content, no meta-commentary

TECHNICAL TERMS TO PRESERVE:
- IPMI, BMC, Redfish commands and responses
- API endpoints and parameters  
- Configuration values and constants
- Error codes and status messages"""

    return "You are a helpful assistant for processing technical documents."

# =======================================================================================
# vLLM ENGINE WRAPPER
# =======================================================================================

class VLLMEngineManager:
    def __init__(self):
        self.engine: Optional[AsyncLLMEngine] = None
        self.current_model: Optional[str] = None
        
    async def initialize_engine(self, model_name: str):
        """Инициализация или переключение модели"""
        if self.current_model == model_name and self.engine is not None:
            return  # Модель уже загружена
            
        logger.info(f"Loading model: {model_name}")
        
        # Останавливаем текущий engine если есть
        if self.engine is not None:
            # В vLLM нет прямого метода остановки, создаем новый instance
            pass
            
        # Настройка параметров engine
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=config.tensor_parallel_size,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=config.trust_remote_code,
            download_dir=config.hf_home
        )
        
        try:
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.current_model = model_name
            model_loaded.labels(model=model_name).set(1)
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            error_counter.labels(model=model_name, error_type="model_loading").inc()
            logger.error(f"Failed to load model {model_name}: {e}")
            raise HTTPException(status_code=500, f"Failed to load model: {e}")
    
    async def generate_response(self, prompt: str, sampling_params: SamplingParams) -> str:
        """Генерация ответа"""
        if self.engine is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        request_id = random_uuid()
        results = self.engine.generate(prompt, sampling_params, request_id)
        
        # Ждем результат
        final_output = None
        async for request_output in results:
            final_output = request_output
            
        if final_output is None:
            raise HTTPException(status_code=500, detail="No output generated")
            
        return final_output.outputs[0].text

engine_manager = VLLMEngineManager()

# =======================================================================================
# FASTAPI APPLICATION
# =======================================================================================

app = FastAPI(
    title="vLLM PDF Converter Server",
    description="OpenAI-compatible API for PDF document processing",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS настройка
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================================================
# API ENDPOINTS
# =======================================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    await update_gpu_metrics()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": engine_manager.current_model,
        "gpu_available": True,  # TODO: реальная проверка GPU
        "memory_usage": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
    }

@app.get("/v1/models")
async def list_models():
    """Список доступных моделей (OpenAI API compatibility)"""
    models = [
        ModelInfo(id=config.text_model),
        ModelInfo(id=config.vision_model)
    ]
    return {"data": models}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint (OpenAI API compatibility)"""
    start_time = time.time()
    active_requests.inc()
    request_counter.labels(model=request.model, endpoint="chat_completions").inc()
    
    try:
        # Инициализируем нужную модель
        await engine_manager.initialize_engine(request.model)
        
        # Формируем промпт из сообщений
        system_message = ""
        user_messages = []
        
        for msg in request.messages:
            if msg.role == "system":
                system_message = msg.content
            elif msg.role == "user":
                user_messages.append(msg.content)
        
        # Объединяем промпт
        if system_message:
            full_prompt = f"System: {system_message}\n\nUser: {' '.join(user_messages)}"
        else:
            full_prompt = ' '.join(user_messages)
        
        # Параметры семплирования
        sampling_params = SamplingParams(
            temperature=request.temperature or 0.1,
            top_p=request.top_p or 0.8,
            top_k=request.top_k or 30,
            max_tokens=request.max_tokens or 2048,
            stop=request.stop
        )
        
        # Генерируем ответ
        response_text = await engine_manager.generate_response(full_prompt, sampling_params)
        
        # Метрики
        duration = time.time() - start_time
        request_duration.labels(model=request.model, endpoint="chat_completions").observe(duration)
        token_generation.labels(model=request.model).observe(len(response_text.split()))
        
        # Формируем OpenAI-совместимый ответ
        response_data = {
            "id": f"chatcmpl-{random_uuid()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(full_prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(full_prompt.split()) + len(response_text.split())
            }
        }
        
        return response_data
        
    except Exception as e:
        error_counter.labels(model=request.model, error_type="generation").inc()
        logger.error(f"Error in chat completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        active_requests.dec()

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completions endpoint (OpenAI API compatibility)"""
    start_time = time.time()
    active_requests.inc()
    request_counter.labels(model=request.model, endpoint="completions").inc()
    
    try:
        await engine_manager.initialize_engine(request.model)
        
        sampling_params = SamplingParams(
            temperature=request.temperature or 0.1,
            top_p=request.top_p or 0.8,
            top_k=request.top_k or 30,
            max_tokens=request.max_tokens or 2048,
            stop=request.stop
        )
        
        response_text = await engine_manager.generate_response(request.prompt, sampling_params)
        
        # Метрики
        duration = time.time() - start_time
        request_duration.labels(model=request.model, endpoint="completions").observe(duration)
        token_generation.labels(model=request.model).observe(len(response_text.split()))
        
        response_data = {
            "id": f"cmpl-{random_uuid()}",
            "object": "text_completion", 
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": response_text,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(request.prompt.split()) + len(response_text.split())
            }
        }
        
        return response_data
        
    except Exception as e:
        error_counter.labels(model=request.model, error_type="generation").inc()
        logger.error(f"Error in completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        active_requests.dec()

# =======================================================================================
# СПЕЦИАЛИЗИРОВАННЫЕ ENDPOINTS ДЛЯ PDF PIPELINE
# =======================================================================================

@app.post("/v1/document/convert")
async def convert_document(request: dict):
    """Специализированный endpoint для конвертации документов"""
    
    # Используем специализированный промпт
    system_prompt = get_specialized_system_prompt("document_conversion")
    
    chat_request = ChatCompletionRequest(
        model=config.vision_model,  # Используем vision модель для документов
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=request.get("content", ""))
        ],
        temperature=0.1,
        max_tokens=4096
    )
    
    return await chat_completions(chat_request)

@app.post("/v1/document/translate")
async def translate_document(request: dict):
    """Специализированный endpoint для перевода"""
    
    source_lang = request.get("source_lang", "zh-CN")
    target_lang = request.get("target_lang", "ru")
    content = request.get("content", "")
    
    system_prompt = get_specialized_system_prompt("translation", source_lang, target_lang)
    
    chat_request = ChatCompletionRequest(
        model=config.text_model,  # Используем текстовую модель для перевода
        messages=[
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=content)
        ],
        temperature=0.1,
        max_tokens=4096
    )
    
    return await chat_completions(chat_request)

# =======================================================================================
# STARTUP & SHUTDOWN
# =======================================================================================

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    logger.info("Starting vLLM PDF Converter Server v4.0")
    
    # Запускаем Prometheus метрики сервер
    start_http_server(8001)
    logger.info("Prometheus metrics server started on port 8001")
    
    # Инициализируем модель по умолчанию
    await engine_manager.initialize_engine(config.text_model)

@app.on_event("shutdown") 
async def shutdown_event():
    """Очистка при завершении"""
    logger.info("Shutting down vLLM server")

# =======================================================================================
# MAIN
# =======================================================================================

if __name__ == "__main__":
    logger.info(f"Starting vLLM server on {config.host}:{config.port}")
    logger.info(f"Text model: {config.text_model}")
    logger.info(f"Vision model: {config.vision_model}")
    logger.info(f"HF cache: {config.hf_home}")
    
    uvicorn.run(
        "vllm_server:app",
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=True
    )