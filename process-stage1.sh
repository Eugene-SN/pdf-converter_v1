#!/bin/bash

# =============================================================================
# PDF CONVERTER PIPELINE v2.0 - ЭТАП 1: КОНВЕРТАЦИЯ В MARKDOWN
# Запускает только DAG 1 (Document Preprocessing) + DAG 2 (Content Transformation)
# Результат: PDF → Markdown (китайский оригинал) без перевода
# =============================================================================

set -euo pipefail

# Конфигурация
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.env"

# Локальные папки для скрипта на хосте
HOST_INPUT_DIR="${SCRIPT_DIR}/input_pdf"
HOST_OUTPUT_DIR="${SCRIPT_DIR}/output_md_zh"
LOGS_DIR="${SCRIPT_DIR}/logs"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Загрузка конфигурации
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# URL сервисов
AIRFLOW_BASE_URL="http://localhost:8090"
AIRFLOW_USERNAME=${AIRFLOW_USERNAME:-"admin"}
AIRFLOW_PASSWORD=${AIRFLOW_PASSWORD:-"admin"}

# Создание директорий
mkdir -p "$HOST_INPUT_DIR" "$HOST_OUTPUT_DIR" "$LOGS_DIR"

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[$timestamp]${NC} ${YELLOW}[$level]${NC} $message" | tee -a "$LOGS_DIR/stage1_$(date +%Y%m%d_%H%M%S).log"
}

show_header() {
    echo -e "${BLUE}"
    echo "==============================================================================="
    echo "  PDF CONVERTER PIPELINE v2.0 - ЭТАП 1: КОНВЕРТАЦИЯ В MARKDOWN"
    echo "==============================================================================="
    echo -e "${NC}"
    echo "🎯 Цель: PDF → Markdown (китайский оригинал)"
    echo "📂 Входная папка: $HOST_INPUT_DIR"
    echo "📁 Выходная папка: $HOST_OUTPUT_DIR"
    echo ""
}

check_services() {
    log "INFO" "Проверка готовности сервисов..."
    
    # Проверка Airflow
    if ! curl -s --user "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" "$AIRFLOW_BASE_URL/health" > /dev/null 2>&1; then
        log "ERROR" "Airflow UI недоступен на $AIRFLOW_BASE_URL"
        log "INFO" "Убедитесь что система запущена: ./start-system.sh"
        exit 1
    fi
    
    # Проверка Document Processor - ИСПРАВЛЕНО: localhost вместо имени контейнера
    local doc_proc_url="http://localhost:8001"
    if ! curl -s "$doc_proc_url/health" > /dev/null 2>&1; then
        log "ERROR" "Document Processor недоступен на $doc_proc_url"
        log "INFO" "Проверьте что контейнер document-processor запущен"
        exit 1
    fi
    
    # Проверка vLLM - ИСПРАВЛЕНО: localhost вместо имени контейнера
    local vllm_url="http://localhost:8000"
    if ! curl -s "$vllm_url/health" > /dev/null 2>&1; then
        log "ERROR" "vLLM Server недоступен на $vllm_url"
        log "INFO" "Проверьте что контейнер vllm-server запущен и загрузил модель"
        exit 1
    fi
    
    log "INFO" "✅ Все необходимые сервисы готовы"
}

trigger_dag() {
    local dag_id="$1"
    local config="$2"
    local description="$3"
    
    log "INFO" "🚀 Запуск $description..."
    
    # Создание JSON конфигурации
    local json_config=$(echo "$config" | python3 -c "
import sys, json
try:
    config_dict = {}
    for line in sys.stdin:
        if '=' in line:
            key, value = line.strip().split('=', 1)
            # Попытка преобразовать в правильный тип
            if value.lower() in ['true', 'false']:
                config_dict[key] = value.lower() == 'true'
            elif value.isdigit():
                config_dict[key] = int(value)
            else:
                config_dict[key] = value
    print(json.dumps(config_dict))
except Exception as e:
    print('{}')
")
    
    # Запуск DAG через API
    local response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        --user "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" \
        -H "Content-Type: application/json" \
        -d "{\"conf\": $json_config}" \
        "$AIRFLOW_BASE_URL/api/v1/dags/$dag_id/dagRuns")
    
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" -eq 200 ] || [ "$http_code" -eq 201 ]; then
        local dag_run_id=$(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('dag_run_id', 'unknown'))" 2>/dev/null || echo "unknown")
        log "INFO" "✅ DAG запущен. Run ID: $dag_run_id"
        echo "$dag_run_id"
    else
        log "ERROR" "❌ Ошибка запуска DAG: HTTP $http_code"
        log "ERROR" "Ответ: $body"
        return 1
    fi
}

wait_for_dag_completion() {
    local dag_id="$1"
    local dag_run_id="$2"
    local description="$3"
    local timeout=${4:-1800}  # 30 минут по умолчанию
    
    log "INFO" "⏳ Ожидание завершения $description (таймаут: ${timeout}s)..."
    
    local start_time=$(date +%s)
    local dots=0
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout ]; then
            log "ERROR" "❌ Таймаут ожидания $description"
            return 1
        fi
        
        # Получение статуса DAG
        local response=$(curl -s \
            --user "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" \
            "$AIRFLOW_BASE_URL/api/v1/dags/$dag_id/dagRuns/$dag_run_id")
        
        local state=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('state', 'unknown'))
except:
    print('error')
" 2>/dev/null || echo "error")
        
        case "$state" in
            "success")
                log "INFO" "✅ $description завершен успешно"
                return 0
                ;;
            "failed"|"upstream_failed")
                log "ERROR" "❌ $description завершен с ошибкой"
                return 1
                ;;
            "running")
                # Показываем прогресс
                dots=$(((dots + 1) % 4))
                local progress_dots=$(printf "%*s" $dots '' | tr ' ' '.')
                printf "\r${YELLOW}[ВЫПОЛНЯЕТСЯ]${NC} $description$progress_dots   "
                sleep 5
                ;;
            *)
                sleep 3
                ;;
        esac
    done
}

process_single_file() {
    local pdf_file="$1"
    local filename=$(basename "$pdf_file")
    local timestamp=$(date +%s)
    
    log "INFO" "📄 Начинаем обработку: $filename"
    
    # Конфигурация для этапа 1 (только конвертация)
    local stage1_config="
        input_file=$pdf_file
        filename=$filename
        timestamp=$timestamp
        target_language=zh
        quality_level=high
        use_ocr=false
        preserve_structure=true
        extract_tables=true
        extract_images=true
        stage_mode=conversion_only
        processing_stages=2"
    
    # Этап 1.1: Document Preprocessing
    local dag1_run_id
    dag1_run_id=$(trigger_dag "document_preprocessing" "$stage1_config" "Document Preprocessing")
    
    if [ $? -eq 0 ]; then
        if wait_for_dag_completion "document_preprocessing" "$dag1_run_id" "Document Preprocessing" 1800; then
            log "INFO" "✅ Этап 1.1 завершен: извлечение контента"
        else
            log "ERROR" "❌ Ошибка в этапе 1.1"
            return 1
        fi
    else
        log "ERROR" "❌ Не удалось запустить Document Preprocessing"
        return 1
    fi
    
    # Этап 1.2: Content Transformation
    local stage2_config="
        intermediate_file=/tmp/dag1_results_${timestamp}.json
        original_config=$stage1_config
        dag1_completed=true
        vllm_model=Qwen/Qwen2.5-VL-32B-Instruct
        transformation_quality=high
        preserve_technical_terms=true"
    
    local dag2_run_id
    dag2_run_id=$(trigger_dag "content_transformation" "$stage2_config" "Content Transformation")
    
    if [ $? -eq 0 ]; then
        if wait_for_dag_completion "content_transformation" "$dag2_run_id" "Content Transformation" 1200; then
            log "INFO" "✅ Этап 1.2 завершен: преобразование в Markdown"
            
            # Проверка результата
            local output_file="$HOST_OUTPUT_DIR/${timestamp}_${filename%.pdf}.md"
            if [ -f "$output_file" ]; then
                log "INFO" "📁 Результат сохранен: $output_file"
                return 0
            else
                log "WARN" "⚠️ Файл результата не найден: $output_file"
                return 1
            fi
        else
            log "ERROR" "❌ Ошибка в этапе 1.2"
            return 1
        fi
    else
        log "ERROR" "❌ Не удалось запустить Content Transformation"
        return 1
    fi
}

process_batch() {
    log "INFO" "🔍 Поиск PDF файлов в $HOST_INPUT_DIR"
    
    # Поиск всех PDF файлов
    local pdf_files=()
    while IFS= read -r -d '' file; do
        pdf_files+=("$file")
    done < <(find "$HOST_INPUT_DIR" -name "*.pdf" -type f -print0)
    
    local total_files=${#pdf_files[@]}
    
    if [ $total_files -eq 0 ]; then
        log "WARN" "📂 Нет PDF файлов в $HOST_INPUT_DIR"
        echo "Поместите PDF файлы в папку $HOST_INPUT_DIR и запустите снова"
        return 0
    fi
    
    log "INFO" "📊 Найдено файлов для обработки: $total_files"
    echo ""
    
    # Обработка файлов
    local processed=0
    local failed=0
    local start_time=$(date +%s)
    
    for pdf_file in "${pdf_files[@]}"; do
        local filename=$(basename "$pdf_file")
        echo -e "${BLUE}[ФАЙЛ $((processed + failed + 1))/$total_files]${NC} $filename"
        
        if process_single_file "$pdf_file"; then
            ((processed++))
            echo -e "Статус: ${GREEN}✅ УСПЕШНО${NC}"
        else
            ((failed++))
            echo -e "Статус: ${RED}❌ ОШИБКА${NC}"
        fi
        
        echo ""
    done
    
    # Итоговая статистика
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    echo "==============================================================================="
    echo -e "${GREEN}ЭТАП 1 ЗАВЕРШЕН: КОНВЕРТАЦИЯ В MARKDOWN${NC}"
    echo "==============================================================================="
    echo -e "📊 Статистика обработки:"
    echo -e "   Успешно обработано: ${GREEN}$processed${NC} файлов"
    echo -e "   Ошибок: ${RED}$failed${NC} файлов"
    echo -e "   Общее время: ${BLUE}$total_duration${NC} секунд"
    echo ""
    echo -e "📁 Результаты сохранены в: ${YELLOW}$HOST_OUTPUT_DIR${NC}"
    echo -e "📋 Логи сохранены в: ${YELLOW}$LOGS_DIR${NC}"
    echo ""
    
    if [ $failed -gt 0 ]; then
        echo -e "${YELLOW}⚠️ Рекомендации:${NC}"
        echo "   - Проверьте логи для диагностики ошибок"
        echo "   - Убедитесь что PDF файлы не повреждены"
        echo "   - Проверьте доступность всех сервисов"
    else
        echo -e "${GREEN}🎉 Все файлы успешно конвертированы в Markdown!${NC}"
        echo ""
        echo "Следующие шаги:"
        echo "   - Для валидации: ./process-stage2.sh"
        echo "   - Для перевода: ./process-stage3.sh [язык]"
    fi
}

# Основная логика
main() {
    show_header
    check_services
    
    echo -e "${YELLOW}Нажмите Enter для начала обработки или Ctrl+C для отмены...${NC}"
    read -r
    
    process_batch
}

# Запуск, если скрипт вызван напрямую
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi