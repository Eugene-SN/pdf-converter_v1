#!/bin/bash
# vLLM Start Script для PDF Converter Pipeline v2.0
# Автоматическая настройка и запуск vLLM сервера

set -e

echo "🚀 Запуск vLLM Server для PDF Converter Pipeline v2.0"

# Конфигурация по умолчанию
DEFAULT_MODEL="${VLLM_MODEL_NAME:-Qwen/Qwen2.5-VL-32B-Instruct}"
MODEL_ALIAS="${VLLM_MODEL_ALIAS:-Qwen2.5-VL-32B-Instruct}"
SERVED_MODEL_NAME="${MODEL_ALIAS}"

# GPU конфигурация
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
BLOCK_SIZE="${VLLM_BLOCK_SIZE:-16}"
SWAP_SPACE="${VLLM_SWAP_SPACE:-4}"

# Generation параметры
TEMPERATURE="${VLLM_TEMPERATURE:-0.1}"
MAX_TOKENS="${VLLM_MAX_TOKENS:-4096}"
TOP_P="${VLLM_TOP_P:-0.9}"
TOP_K="${VLLM_TOP_K:-50}"

# Сервер настройки
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${VLLM_API_KEY:-pdf-converter-secure-key-2024}"

# Дополнительные настройки
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"

echo "📋 Конфигурация vLLM:"
echo "   Модель: $DEFAULT_MODEL"
echo "   Алиас: $MODEL_ALIAS"
echo "   Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo "   GPU Memory: $GPU_MEMORY_UTILIZATION"
echo "   Max Model Length: $MAX_MODEL_LEN"
echo "   Host: $HOST:$PORT"

# Проверка GPU
echo "🔍 Проверка GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ GPU обнаружены"
else
    echo "⚠️ nvidia-smi не найден, продолжаем без GPU проверки"
fi

# Создание директорий
mkdir -p /models/huggingface /app/logs

# Проверка переменных окружения
if [ -z "$HF_HOME" ]; then
    export HF_HOME="/models/huggingface"
fi

echo "🏠 HuggingFace Home: $HF_HOME"

# Ожидание готовности модели (если нужна загрузка)
echo "⏳ Подготовка модели $DEFAULT_MODEL..."

# Запуск vLLM сервера
echo "🚀 Запуск vLLM OpenAI-совместимого сервера..."

exec python -m vllm.entrypoints.openai.api_server \
    --model "$DEFAULT_MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --block-size "$BLOCK_SIZE" \
    --swap-space "$SWAP_SPACE" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --enable-prefix-caching \
    --api-key "$API_KEY" \
    --disable-log-stats