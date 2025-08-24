#!/bin/bash

# =============================================================================
# PDF CONVERTER PIPELINE v2.0 - ЭТАП 1: КОНВЕРТАЦИЯ В MARKDOWN
# Запускает только DAG 1 (Document Preprocessing) + DAG 2 (Content Transformation)
# Результат: PDF → Markdown (китайский оригинал) без перевода
# =============================================================================

set -euo pipefail

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.env"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Конфигурация сервисов (берем из env HOST переменных; fallback на прежние дефолты)
AIRFLOW_URL="${AIRFLOW_BASE_URL_HOST:-http://localhost:8090}"
DOCUMENT_PROCESSOR_URL="${DOCUMENT_PROCESSOR_URL_HOST:-http://localhost:8001}"
VLLM_URL="${VLLM_SERVER_URL_HOST:-http://localhost:8000}"
QA_URL="${QUALITY_ASSURANCE_URL_HOST:-http://localhost:8002}"
TRANSLATOR_URL="${TRANSLATOR_URL_HOST:-http://localhost:8003}"

AIRFLOW_USERNAME="${AIRFLOW_USERNAME:-admin}"
AIRFLOW_PASSWORD="${AIRFLOW_PASSWORD:-admin}"

# Параметры по умолчанию
DEFAULT_TARGET_LANGUAGE=""         # Целевой язык на этапе препроцессинга НЕ используется
DEFAULT_QUALITY_LEVEL="high"
DEFAULT_USE_OCR=false # ✅ ИСПРАВЛЕНО: По умолчанию OCR отключен для цифровых PDF

# Локальные папки для скрипта на хосте
HOST_INPUT_DIR="${SCRIPT_DIR}/input_pdf"
HOST_OUTPUT_DIR="${SCRIPT_DIR}/output_md_zh"
LOGS_DIR="${SCRIPT_DIR}/logs"

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

print_header() {
  echo -e "${BLUE}=================================="
  echo -e "✅ ИСПРАВЛЕННЫЙ PDF CONVERTER v4.0"
  echo -e "Stage 1: Document Preprocessing"
  echo -e "==================================${NC}"
}

print_error() {
  echo -e "${RED}❌ ОШИБКА: $1${NC}"
}

print_success() {
  echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
  echo -e "${YELLOW}⚠️ $1${NC}"
}

print_info() {
  echo -e "${BLUE}ℹ️ $1${NC}"
}

# Проверка доступности сервиса
check_service() {
  local service_url=$1
  local service_name=$2
  local timeout=10
  print_info "Проверяем доступность $service_name..."
  if curl -s --max-time $timeout "$service_url/health" > /dev/null 2>&1; then
    print_success "$service_name доступен"
    return 0
  else
    print_error "$service_name недоступен по адресу $service_url"
    return 1
  fi
}

# ✅ ИСПРАВЛЕНО: Проверка состояния Document Processor
check_document_processor() {
  print_info "Проверяем состояние Document Processor..."
  local response
  response=$(curl -s "$DOCUMENT_PROCESSOR_URL/status" 2>/dev/null || echo "")
  if [ -n "$response" ]; then
    # теперь docling может быть строкой "official_api"
    local docling_status
    docling_status=$(echo "$response" | jq -r '.processors.docling // ""' 2>/dev/null || echo "")
    local ocr_initialized
    ocr_initialized=$(echo "$response" | jq -r '.processors.ocr // false' 2>/dev/null || echo "false")

    print_info "Статус Document Processor:"
    print_info "  - Docling: $docling_status"
    print_info "  - OCR инициализирован: $ocr_initialized"

    # считаем здоровым любой непустой docling_status, кроме "unavailable" или "false"
    if [[ "$docling_status" != "" && "$docling_status" != "false" && "$docling_status" != "unavailable" ]]; then
      print_success "Document Processor готов к работе"
      return 0
    else
      print_warning "Document Processor не полностью готов"
      return 1
    fi
  else
    print_error "Не удалось получить статус Document Processor"
    return 1
  fi
}

# Проверка готовности Airflow
check_airflow() {
  print_info "Проверяем доступность Airflow..."
  # Проверяем health endpoint
  if ! check_service "$AIRFLOW_URL" "Airflow"; then
    return 1
  fi
  # Проверяем доступность API
  local auth_header
  auth_header="Authorization: Basic $(echo -n "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" | base64)"
  local api_response
  api_response=$(curl -s -H "$auth_header" "$AIRFLOW_URL/api/v1/dags" 2>/dev/null || echo "")
  if [ -n "$api_response" ]; then
    print_success "Airflow API доступен"
    return 0
  else
    print_error "Airflow API недоступен"
    return 1
  fi
}

# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ
# =============================================================================

show_usage() {
  echo "Использование: $0 [options] [input_file]"
  echo ""
  echo "Параметры:"
  echo "  input_file                 Путь к PDF файлу для обработки (не обязателен; иначе будет предложен выбор из ${HOST_INPUT_DIR})"
  echo ""
  echo "Опции:"
  echo "  --target-language LANG     [НЕ ИСПОЛЬЗУЕТСЯ НА ЭТОМ ЭТАПЕ] Целевой язык задается только на этапе перевода"
  echo "  --quality-level LEVEL      Уровень качества (high, medium, fast). По умолчанию: $DEFAULT_QUALITY_LEVEL"
  echo "  --use-ocr                  Включить OCR для сканированных PDF"
  echo "  --no-ocr                   Отключить OCR (по умолчанию для цифровых PDF)"
  echo "  --extract-tables           Извлекать таблицы (включено по умолчанию)"
  echo "  --no-extract-tables        Отключить извлечение таблиц"
  echo "  --extract-images           Извлекать изображения (включено по умолчанию)"
  echo "  --no-extract-images        Отключить извлечение изображений"
  echo "  --batch                    Пакетная обработка всех PDF из ${HOST_INPUT_DIR}"
  echo "  --help                     Показать справку"
  echo ""
  echo "Примеры:"
  echo "  $0"
  echo "  $0 --no-ocr --quality-level high"
  echo "  $0 /path/to/scanned.pdf --use-ocr"
}

list_input_dir_files() {
  if [ ! -d "$HOST_INPUT_DIR" ]; then
    print_error "Каталог не найден: $HOST_INPUT_DIR"
    exit 1
  fi
  mapfile -t PDF_FILES < <(find "$HOST_INPUT_DIR" -maxdepth 1 -type f -name "*.pdf" | sort)
  if [ ${#PDF_FILES[@]} -eq 0 ]; then
    print_error "В каталоге $HOST_INPUT_DIR нет PDF файлов"
    exit 1
  fi
}

prompt_select_file() {
  list_input_dir_files
  echo "Доступные PDF файлы в $HOST_INPUT_DIR:"
  local idx=1
  for f in "${PDF_FILES[@]}"; do
    echo "  [$idx] $(basename "$f")"
    idx=$((idx+1))
  done
  echo -n "Выберите номер файла для обработки: "
  read -r sel
  if ! [[ "$sel" =~ ^[0-9]+$ ]]; then
    print_error "Некорректный номер"
    exit 1
  fi
  if [ "$sel" -lt 1 ] || [ "$sel" -gt ${#PDF_FILES[@]} ]; then
    print_error "Номер вне диапазона"
    exit 1
  fi
  INPUT_FILE="${PDF_FILES[$((sel-1))]}"
}

# Парсинг аргументов командной строки
parse_arguments() {
  INPUT_FILE=""
  TARGET_LANGUAGE="$DEFAULT_TARGET_LANGUAGE" # не используется на этом этапе
  QUALITY_LEVEL="$DEFAULT_QUALITY_LEVEL"
  USE_OCR="$DEFAULT_USE_OCR" # ✅ ИСПРАВЛЕНО: По умолчанию false
  EXTRACT_TABLES=true
  EXTRACT_IMAGES=true
  BATCH_MODE=false

  while [[ $# -gt 0 ]]; do
    case $1 in
      --help)
        show_usage
        exit 0
        ;;
      --target-language)
        TARGET_LANGUAGE="$2"
        shift 2
        ;;
      --quality-level)
        QUALITY_LEVEL="$2"
        shift 2
        ;;
      --use-ocr)
        USE_OCR=true
        shift
        ;;
      --no-ocr)
        USE_OCR=false
        shift
        ;;
      --extract-tables)
        EXTRACT_TABLES=true
        shift
        ;;
      --no-extract-tables)
        EXTRACT_TABLES=false
        shift
        ;;
      --extract-images)
        EXTRACT_IMAGES=true
        shift
        ;;
      --no-extract-images)
        EXTRACT_IMAGES=false
        shift
        ;;
      --batch)
        BATCH_MODE=true
        shift
        ;;
      -*)
        print_error "Неизвестная опция: $1"
        show_usage
        exit 1
        ;;
      *)
        if [ -z "$INPUT_FILE" ]; then
          INPUT_FILE="$1"
        else
          print_error "Слишком много аргументов: $1"
          show_usage
          exit 1
        fi
        shift
        ;;
    esac
  done

  # Если пакетный режим - не проверяем одиночный INPUT_FILE
  if [ "$BATCH_MODE" = true ]; then
    return 0
  fi

  # Если не передан файл - предложить выбрать из HOST_INPUT_DIR
  if [ -z "$INPUT_FILE" ]; then
    prompt_select_file
  fi

  # Проверяем существование файла
  if [ ! -f "$INPUT_FILE" ]; then
    print_error "Файл не найден: $INPUT_FILE"
    exit 1
  fi

  # Проверяем что файл PDF
  if [[ ! "$INPUT_FILE" =~ \.pdf$ ]]; then
    print_error "Файл должен иметь расширение .pdf: $INPUT_FILE"
    exit 1
  fi
}

# Валидация параметров
validate_parameters() {
  print_info "Валидация параметров..."
  # Уровень качества
  case "$QUALITY_LEVEL" in
    high|medium|fast)
      print_success "Уровень качества: $QUALITY_LEVEL"
      ;;
    *)
      print_error "Неподдерживаемый уровень качества: $QUALITY_LEVEL"
      print_info "Поддерживаемые уровни: high, medium, fast"
      exit 1
      ;;
  esac

  # Если одиночный режим - проверка размера файла
  if [ -n "$INPUT_FILE" ] && [ "$BATCH_MODE" = false ]; then
    local file_size
    file_size=$(stat -f%z "$INPUT_FILE" 2>/dev/null || stat -c%s "$INPUT_FILE" 2>/dev/null || echo "0")
    local max_size=$((500 * 1024 * 1024)) # 500MB
    if [ "$file_size" -gt "$max_size" ]; then
      print_error "Файл слишком большой: ${file_size} байт (максимум: ${max_size})"
      exit 1
    fi
    print_success "Размер файла: $(echo "$file_size" | awk '{printf "%.2f MB", $1/1024/1024}')"
  fi
}

# ✅ ИСПРАВЛЕНО: Формирование корректного payload и запуск DAG
trigger_single_file() {
  local file_path="$1"
  local abs_input_file filename timestamp auth_header api_url dag_config response dag_run_id

  print_info "Запуск DAG document_preprocessing..."

  abs_input_file=$(realpath "$file_path")
  filename=$(basename "$file_path")
  timestamp=$(date +%s)

  # Конфигурация conf для DAG 1 (без target_language на этом этапе)
  # Минимальный набор: input_file, filename, timestamp, use_ocr, extract_* и т.п. —
  # но передаем всё через conf, как ожидает Airflow v1 API.
  dag_config=$(jq -n \
    --arg input_file "$abs_input_file" \
    --arg filename "$filename" \
    --argjson use_ocr $USE_OCR \
    --arg quality_level "$QUALITY_LEVEL" \
    --argjson extract_tables $EXTRACT_TABLES \
    --argjson extract_images $EXTRACT_IMAGES \
    --arg language "zh-CN" \
    --argjson timestamp $timestamp \
    '{
      conf: {
        input_file: $input_file,
        filename: $filename,
        use_ocr: $use_ocr,
        quality_level: $quality_level,
        extract_tables: $extract_tables,
        extract_images: $extract_images,
        extract_formulas: true,
        high_quality_ocr: true,
        preserve_structure: true,
        timestamp: $timestamp,
        language: $language,
        chinese_optimization: true,
        pipeline_version: "4.0",
        processing_mode: "digital_pdf"
      }
    }'
  )

  print_info "Конфигурация DAG:"
  echo "$dag_config" | jq '.' 2>/dev/null || echo "$dag_config"

  auth_header="Authorization: Basic $(echo -n "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" | base64)"
  api_url="$AIRFLOW_URL/api/v1/dags/document_preprocessing/dagRuns"

  print_info "Отправляем запрос к Airflow API..."
  response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -H "$auth_header" \
    -d "$dag_config" \
    "$api_url" 2>/dev/null || echo "")

  if [ -n "$response" ]; then
    dag_run_id=$(echo "$response" | jq -r '.dag_run_id // empty' 2>/dev/null || echo "")
    if [ -n "$dag_run_id" ]; then
      print_success "DAG запущен успешно!"
      print_info "DAG Run ID: $dag_run_id"
      local monitoring_url="$AIRFLOW_URL/dags/document_preprocessing/grid?dag_run_id=$dag_run_id"
      print_info "Мониторинг: $monitoring_url"
      return 0
    else
      print_error "Не удалось получить DAG Run ID из ответа"
      print_info "Ответ API: $response"
      return 1
    fi
  else
    print_error "Не удалось запустить DAG - пустой ответ от API"
    return 1
  fi
}

trigger_batch() {
  list_input_dir_files
  local failures=0
  for f in "${PDF_FILES[@]}"; do
    print_info "-----"
    print_info "Обработка файла: $(basename "$f")"
    if ! trigger_single_file "$f"; then
      failures=$((failures+1))
      print_warning "Файл завершился с ошибкой запуска: $(basename "$f")"
    fi
  done
  if [ "$failures" -gt 0 ]; then
    print_warning "Пакетная обработка завершена с ошибками: $failures"
    return 1
  else
    print_success "Пакетная обработка завершена успешно"
    return 0
  fi
}

# Мониторинг выполнения DAG
monitor_dag_execution() {
  local dag_run_id="$1"
  local timeout=1800 # 30 минут
  local interval=10 # Проверяем каждые 10 секунд
  local elapsed=0
  print_info "Мониторинг выполнения DAG (timeout: ${timeout}s)..."
  local auth_header="Authorization: Basic $(echo -n "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" | base64)"
  while [ $elapsed -lt $timeout ]; do
    local api_url="$AIRFLOW_URL/api/v1/dags/document_preprocessing/dagRuns/$dag_run_id"
    local response
    response=$(curl -s -H "$auth_header" "$api_url" 2>/dev/null || echo "")
    if [ -n "$response" ]; then
      local state
      state=$(echo "$response" | jq -r '.state // "unknown"' 2>/dev/null || echo "unknown")
      case "$state" in
        "success")
          print_success "DAG выполнен успешно!"
          return 0
          ;;
        "failed")
          print_error "DAG завершился с ошибкой"
          return 1
          ;;
        "running")
          print_info "DAG выполняется... (${elapsed}s)"
          ;;
        *)
          print_info "Состояние DAG: $state (${elapsed}s)"
          ;;
      esac
    else
      print_warning "Не удалось получить статус DAG"
    fi
    sleep $interval
    elapsed=$((elapsed + interval))
  done
  print_error "Timeout мониторинга DAG (${timeout}s)"
  return 1
}

# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

main() {
  print_header

  # Парсим аргументы
  parse_arguments "$@"

  # Показываем конфигурацию
  print_info "Конфигурация обработки:"
  if [ "$BATCH_MODE" = true ]; then
    print_info "  - Режим: пакетная обработка папки ${HOST_INPUT_DIR}"
  else
    print_info "  - Входной файл: ${INPUT_FILE}"
  fi
  # На этом этапе целевой язык не задается — перевод позже
  print_info "  - Уровень качества: $QUALITY_LEVEL"
  print_info "  - Использовать OCR: $USE_OCR" # ✅ ИСПРАВЛЕНО: Показываем правильное значение
  print_info "  - Извлекать таблицы: $EXTRACT_TABLES"
  print_info "  - Извлекать изображения: $EXTRACT_IMAGES"

  # Валидация параметров
  validate_parameters

  # Проверяем доступность сервисов
  print_info "Проверка состояния сервисов..."

  if ! check_airflow; then
    print_error "Airflow недоступен. Проверьте что сервисы запущены: docker-compose up -d"
    exit 1
  fi

  if ! check_document_processor; then
    print_error "Document Processor недоступен или не готов"
    print_info "Проверьте логи: docker logs document-processor"
    exit 1
  fi

  print_success "Все сервисы готовы к работе"

  # Запускаем DAG
  if [ "$BATCH_MODE" = true ]; then
    if trigger_batch; then
      print_success "✅ STAGE 1 (batch) ЗАПУЩЕН УСПЕШНО!"
      print_info ""
      print_info "Следующие шаги:"
      print_info "1. Мониторить прогресс в Airflow UI: $AIRFLOW_URL"
      print_info "2. После завершения запустить Stage 2: ./process-stage2.sh"
      print_info "3. Проверить логи: docker logs document-processor"
    else
      print_error "Не удалось запустить Stage 1 (batch)"
      exit 1
    fi
  else
    if trigger_single_file "$INPUT_FILE"; then
      print_success "✅ STAGE 1 ЗАПУЩЕН УСПЕШНО!"
      print_info ""
      print_info "Следующие шаги:"
      print_info "1. Мониторить прогресс в Airflow UI: $AIRFLOW_URL"
      print_info "2. После завершения запустить Stage 2: ./process-stage2.sh"
      print_info "3. Проверить логи: docker logs document-processor"
    else
      print_error "Не удалось запустить Stage 1"
      exit 1
    fi
  fi
}

# Запускаем главную функцию
main "$@"
