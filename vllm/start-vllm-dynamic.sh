#!/bin/bash
# Обновленный скрипт запуска vLLM с динамической подгрузкой моделей
# PDF Converter Pipeline v2.0 - ИСПРАВЛЕНО

set -e

if [ -z "$VLLM_TENSOR_PARALLEL_SIZE" ] || [ -z "$VLLM_PIPELINE_PARALLEL_SIZE" ]; then
  echo "❌ Parallel sizes not set"
  exit 1
fi

echo "🚀 Запуск Dynamic vLLM Server для PDF Converter Pipeline v2.0"
echo "======================================================================"

# Конфигурация по умолчанию
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${VLLM_API_KEY:-pdf-converter-secure-key-2024}"

# GPU конфигурация для 2x A6000
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"

# Модели для PDF Converter Pipeline
CONTENT_MODEL="${VLLM_MODEL_NAME:-Qwen/Qwen2.5-VL-32B-Instruct}"
TRANSLATION_MODEL="${VLLM_TRANSLATION_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

# Пути
HF_HOME="${HF_HOME:-/models/huggingface}"
LOG_DIR="/app/logs"

echo "📋 Конфигурация Dynamic vLLM Server:"
echo " Content Model: $CONTENT_MODEL"
echo " Translation Model: $TRANSLATION_MODEL" 
echo " Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo " GPU Memory Util: $GPU_MEMORY_UTILIZATION"
echo " Host: $HOST:$PORT"
echo " HuggingFace Cache: $HF_HOME"

# Создание необходимых директорий
echo "📁 Создание директорий..."
mkdir -p "$HF_HOME" "$LOG_DIR" /workspace/temp

# Проверка GPU
echo "🔍 Проверка GPU конфигурации..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits
    
    # Проверяем количество GPU
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    echo "Обнаружено GPU: $GPU_COUNT"
    
    if [ "$GPU_COUNT" -lt 2 ]; then
        echo "⚠️ ВНИМАНИЕ: Обнаружено меньше 2 GPU. Tensor Parallel может не работать оптимально."
        TENSOR_PARALLEL_SIZE=1
    fi
else
    echo "⚠️ nvidia-smi недоступен. Продолжаем без проверки GPU."
fi

# Установка переменных окружения для HuggingFace
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"

# Переменные для CUDA оптимизации
export CUDA_VISIBLE_DEVICES="0,1"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# vLLM специфичные переменные
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_MODELSCOPE=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Переменные для Model Manager
export VLLM_CONTENT_MODEL="$CONTENT_MODEL"
export VLLM_TRANSLATION_MODEL="$TRANSLATION_MODEL"
export VLLM_TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE"
export VLLM_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"

echo "🔧 Проверка доступности моделей..."

# Функция проверки модели в HuggingFace cache
check_model_availability() {
    local model_name=$1
    local model_dir="$HF_HOME/models--${model_name//\/--}"
    
    if [ -d "$model_dir" ]; then
        echo "✅ Модель $model_name найдена в кэше"
        return 0
    else
        echo "📥 Модель $model_name не найдена в кэше, будет загружена при первом использовании"
        return 1
    fi
}

# Проверяем наличие моделей
echo "Проверяем Content Transformation модель..."
check_model_availability "$CONTENT_MODEL"

echo "Проверяем Translation модель..."
check_model_availability "$TRANSLATION_MODEL"

# Проверка доступности Python модулей
echo "🐍 Проверка Python зависимостей..."
python -c "
try:
    import vllm
    print('✅ vLLM:', vllm.__version__)
except ImportError:
    print('❌ vLLM не установлен')
    exit(1)

try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('✅ CUDA доступна:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('✅ CUDA устройств:', torch.cuda.device_count())
except ImportError:
    print('❌ PyTorch не установлен')
    exit(1)

try:
    import fastapi
    print('✅ FastAPI:', fastapi.__version__)
except ImportError:
    print('❌ FastAPI не установлен')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Ошибка проверки зависимостей"
    exit 1
fi

# Функция проверки портов
check_port() {
    local port=$1
    if netstat -tuln | grep ":$port " > /dev/null 2>&1; then
        echo "⚠️ Порт $port уже используется"
        return 1
    else
        echo "✅ Порт $port свободен"
        return 0
    fi
}

echo "🔌 Проверка портов..."
check_port "$PORT"

# Создание файла конфигурации для Model Manager
cat > /workspace/model_config.json << EOF
{
    "models": {
        "content_transformation": {
            "name": "$CONTENT_MODEL",
            "alias": "content-transformer",
            "task_type": "content_transformation",
            "estimated_vram_gb": 32.0,
            "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
            "max_model_len": 8192,
            "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION
        },
        "translation": {
            "name": "$TRANSLATION_MODEL", 
            "alias": "translator",
            "task_type": "translation",
            "estimated_vram_gb": 30.0,
            "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
            "max_model_len": 8192,
            "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION
        }
    },
    "server": {
        "host": "$HOST",
        "port": $PORT,
        "api_key": "$API_KEY"
    },
    "hardware": {
        "gpu_count": 2,
        "total_vram_gb": 96,
        "gpu_type": "NVIDIA A6000"
    }
}
EOF

echo "💾 Конфигурация сохранена в /workspace/model_config.json"

# Настройка логирования
LOG_FILE="$LOG_DIR/vllm_dynamic_server.log"
touch "$LOG_FILE"

# Функция для graceful shutdown
cleanup() {
    echo "🔄 Получен сигнал остановки..."
    if [ ! -z "$VLLM_PID" ]; then
        echo "Остановка vLLM сервера (PID: $VLLM_PID)"
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    echo "✅ Dynamic vLLM Server остановлен"
    exit 0
}

# Регистрация обработчиков сигналов
trap cleanup SIGTERM SIGINT SIGQUIT

echo "🚀 Запуск Dynamic vLLM Server с Model Manager..."
echo "Логи сохраняются в: $LOG_FILE"
echo "======================================================================"

# Установка PYTHONPATH для импорта наших модулей
export PYTHONPATH="/workspace:$PYTHONPATH"

# Запуск нашего динамического сервера вместо стандартного vLLM
cd /workspace

# Проверяем наличие наших файлов
if [ ! -f "model_manager.py" ]; then
    echo "❌ Файл model_manager.py не найден!"
    exit 1
fi

if [ ! -f "dynamic_server.py" ]; then
    echo "❌ Файл dynamic_server.py не найден!"
    exit 1
fi

# Запуск динамического сервера
echo "▶️ Запуск Python Dynamic Server..."
python -m vllm.entrypoints.openai.api_server \
    --model="$CONTENT_MODEL" \
    --host="$HOST" \
    --port="$PORT" \
    --served-model-name="dynamic-model" \
    --disable-log-requests \
    --tensor-parallel-size="$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size="${VLLM_PIPELINE_PARALLEL_SIZE:-1}" \
    --gpu-memory-utilization="$GPU_MEMORY_UTILIZATION" \
    --max-model-len=8192 \
    --enable-prefix-caching \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-num-batched-tokens=2048 \
    --max-num-seqs=32 \
    --dtype="auto" \
    --api-key="$API_KEY" \
    2>&1 | tee "$LOG_FILE" &

VLLM_PID=$!

echo "📊 Dynamic vLLM Server запущен с PID: $VLLM_PID"

# Ожидание готовности сервера
echo "⏳ Ожидание готовности сервера..."
TIMEOUT=300  # 5 минут на инициализацию
ELAPSED=0
HEALTH_URL="http://localhost:$PORT/health"
MODELS_URL="http://localhost:$PORT/v1/models"

while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -s -f "$HEALTH_URL" > /dev/null 2>&1; then
        echo "✅ Dynamic vLLM Server готов к работе!"
        echo "🌐 Health check: $HEALTH_URL"
  
        # Проверка доступных моделей
        echo "📋 Проверка доступных моделей..."
        MODELS_RESPONSE=$(curl -s "http://localhost:$PORT/v1/models" 2>/dev/null || echo "")
        if [ -n "$MODELS_RESPONSE" ]; then
            MODELS_COUNT=$(echo "$MODELS_RESPONSE" | python -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('data', [])))" 2>/dev/null || echo "0")
            echo "📊 Загружено моделей: $MODELS_COUNT"
            echo "📋 Models API: http://localhost:$PORT/v1/models"
        else
            echo "⚠️ Не удалось получить список моделей"
        fi
  
  echo "💬 Chat API: http://localhost:$PORT/v1/chat/completions"
  echo "📊 Status API: http://localhost:$PORT/v1/models/status"
  break
fi
    
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "❌ Dynamic vLLM Server завершился неожиданно"
        exit 1
    fi
    
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "⏳ Ожидание... ($ELAPSED/$TIMEOUT секунд)"
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "❌ Тайм-аут ожидания готовности сервера"
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    exit 1
fi

# Вывод информации о системе
echo ""
echo "🎯 СИСТЕМА ГОТОВА К РАБОТЕ!"
echo "======================================================================"
echo "Dynamic vLLM Server для PDF Converter Pipeline v2.0"
echo ""
echo "📋 Доступные модели:"
echo "  - Content Transformation: $CONTENT_MODEL"
echo "  - Translation: $TRANSLATION_MODEL"
echo ""
echo "🔧 Конфигурация:"
echo "  - GPU: 2x A6000 (48GB VRAM total)"
echo "  - Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo "  - Память GPU: ${GPU_MEMORY_UTILIZATION}0%"
echo ""
echo "🌐 API Endpoints:"
echo "  - Health: http://localhost:$PORT/health"
echo "  - Models: http://localhost:$PORT/v1/models"
echo "  - Chat: http://localhost:$PORT/v1/chat/completions"
echo "  - Status: http://localhost:$PORT/v1/models/status"
echo "  - Metrics: http://localhost:$PORT/metrics"
echo ""
echo "📊 Мониторинг:"
echo "  - Логи: $LOG_FILE"
echo "  - PID: $VLLM_PID"
echo "======================================================================"

# Ожидание завершения процесса
wait "$VLLM_PID"