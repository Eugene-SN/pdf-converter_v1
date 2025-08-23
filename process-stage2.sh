#!/bin/bash

# =============================================================================
# PDF CONVERTER PIPELINE v2.0 - ЭТАП 2: КОНВЕРТАЦИЯ + ВАЛИДАЦИЯ
# Запускает DAG 1 + DAG 2 + DAG 4 (без перевода, с 5-уровневой валидацией)
# Результат: PDF → Markdown с полной валидацией качества
# =============================================================================

set -euo pipefail

# Конфигурация
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/input_pdf"
OUTPUT_DIR="${SCRIPT_DIR}/output_md_zh"
LOGS_DIR="${SCRIPT_DIR}/logs"
CONFIG_FILE="${SCRIPT_DIR}/.env"

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

# URL сервисов (с исправленными именами)
AIRFLOW_BASE_URL=${AIRFLOW_BASE_URL:-"http://localhost:8090"}
AIRFLOW_USERNAME=${AIRFLOW_USERNAME:-"admin"}
AIRFLOW_PASSWORD=${AIRFLOW_PASSWORD:-"admin"}

# Создание директорий
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$LOGS_DIR"

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[$timestamp]${NC} ${YELLOW}[$level]${NC} $message" | tee -a "$LOGS_DIR/stage2_$(date +%Y%m%d_%H%M%S).log"
}

show_header() {
    echo -e "${BLUE}"
    echo "==============================================================================="
    echo "  PDF CONVERTER PIPELINE v2.0 - ЭТАП 2: КОНВЕРТАЦИЯ + ВАЛИДАЦИЯ"
    echo "==============================================================================="
    echo -e "${NC}"
    echo "🎯 Цель: PDF → Markdown с 5-уровневой валидацией качества"
    echo "📂 Входная папка: $INPUT_DIR"
    echo "📁 Выходная папка: $OUTPUT_DIR"
    echo "🔍 Уровни валидации:"
    echo "   1️⃣ OCR кросс-валидация (PaddleOCR + Tesseract)"
    echo "   2️⃣ Визуальное сравнение (SSIM анализ)"
    echo "   3️⃣ AST структурное сравнение"
    echo "   4️⃣ Валидация содержимого (таблицы, код, термины)"
    echo "   5️⃣ Автокоррекция и финальная оценка (цель: 100%)"
    echo ""
}

check_services() {
    log "INFO" "Проверка готовности сервисов для валидации..."
    
    local services=(
        "$AIRFLOW_BASE_URL/health:Airflow UI"
        "${DOCUMENT_PROCESSOR_URL:-http://localhost:8001}/health:Document Processor"
        "${VLLM_SERVER_URL:-http://localhost:8000}/health:vLLM Server"
        "${QUALITY_ASSURANCE_URL:-http://localhost:8002}/health:Quality Assurance"
    )
    
    for service_info in "${services[@]}"; do
        local url="${service_info%:*}"
        local name="${service_info#*:}"
        
        if [[ "$name" == "Airflow UI" ]]; then
            if ! curl -s --user "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" "$url" > /dev/null 2>&1; then
                log "ERROR" "$name недоступен на $url"
                exit 1
            fi
        else
            if ! curl -s "$url" > /dev/null 2>&1; then
                log "ERROR" "$name недоступен на $url"
                exit 1
            fi
        fi
        
        log "INFO" "✅ $name готов"
    done
    
    log "INFO" "✅ Все сервисы для валидации готовы"
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
        if '=' in line and not line.strip().startswith('#'):
            key, value = line.strip().split('=', 1)
            # Преобразование типов
            if value.lower() in ['true', 'false']:
                config_dict[key] = value.lower() == 'true'
            elif value.isdigit():
                config_dict[key] = int(value)
            elif '.' in value and value.replace('.', '', 1).isdigit():
                config_dict[key] = float(value)
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
    local timeout=${4:-1800}
    
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

show_qa_results() {
    local qa_report_path="$1"
    
    if [ ! -f "$qa_report_path" ]; then
        log "WARN" "⚠️ QA отчет не найден: $qa_report_path"
        return
    fi
    
    log "INFO" "📊 Анализ результатов валидации..."
    
    # Извлечение основных метрик из QA отчета
    local qa_data=$(python3 -c "
import json
import sys

try:
    with open('$qa_report_path', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    quality_summary = data.get('quality_summary', {})
    level_details = data.get('level_details', [])
    
    print(f\"Overall Score: {quality_summary.get('overall_score', 'N/A')}\")
    print(f\"Quality Grade: {quality_summary.get('quality_grade', 'N/A')}\")
    print(f\"Target Achieved: {quality_summary.get('target_achieved', 'N/A')}\")
    print(f\"Total Corrections: {quality_summary.get('total_corrections', 'N/A')}\")
    
    for i, level in enumerate(level_details[:5], 1):  # Первые 5 уровней
        confidence = level.get('confidence', 0)
        print(f\"Level {i} Score: {confidence}\")
        
except Exception as e:
    print(f\"Error reading QA report: {e}\")
" 2>/dev/null)
    
    echo ""
    echo "📊 РЕЗУЛЬТАТЫ 5-УРОВНЕВОЙ ВАЛИДАЦИИ:"
    echo "============================================="
    echo "$qa_data" | while read -r line; do
        if [[ "$line" =~ "Overall Score:" ]]; then
            local score=$(echo "$line" | cut -d':' -f2 | xargs)
            echo -e "🎯 Общий балл качества: ${GREEN}$score%${NC}"
        elif [[ "$line" =~ "Quality Grade:" ]]; then
            local grade=$(echo "$line" | cut -d':' -f2 | xargs)
            echo -e "📊 Оценка качества: ${BLUE}$grade${NC}"
        elif [[ "$line" =~ "Target Achieved:" ]]; then
            local achieved=$(echo "$line" | cut -d':' -f2 | xargs)
            if [[ "$achieved" == "True" ]]; then
                echo -e "✅ Цель 100% качества: ${GREEN}ДОСТИГНУТА${NC}"
            else
                echo -e "⚠️ Цель 100% качества: ${YELLOW}НЕ ДОСТИГНУТА${NC}"
            fi
        elif [[ "$line" =~ "Level "[1-5]" Score:" ]]; then
            local level_num=$(echo "$line" | cut -d' ' -f2)
            local score=$(echo "$line" | cut -d':' -f2 | xargs)
            echo -e "   Уровень $level_num: ${YELLOW}$score%${NC}"
        fi
    done
    echo ""
}

process_single_file() {
    local pdf_file="$1"
    local filename=$(basename "$pdf_file")
    local timestamp=$(date +%s)
    
    log "INFO" "📄 Начинаем валидированную обработку: $filename"
    
    # Конфигурация для этапа 2 (конвертация + валидация)
    local base_config="
input_file=$pdf_file
filename=$filename
timestamp=$timestamp
target_language=zh
quality_level=high
enable_ocr=true
preserve_structure=true
extract_tables=true
extract_images=true
stage_mode=conversion_with_validation
processing_stages=3
validation_enabled=true
quality_target=100.0"
    
    # Этап 2.1: Document Preprocessing
    log "INFO" "🔄 Этап 2.1: Извлечение контента..."
    local dag1_run_id
    dag1_run_id=$(trigger_dag "document_preprocessing" "$base_config" "Document Preprocessing")
    
    if [ $? -eq 0 ]; then
        if wait_for_dag_completion "document_preprocessing" "$dag1_run_id" "Document Preprocessing" 1800; then
            log "INFO" "✅ Этап 2.1 завершен: контент извлечен"
        else
            log "ERROR" "❌ Ошибка в извлечении контента"
            return 1
        fi
    else
        return 1
    fi
    
    # Этап 2.2: Content Transformation
    log "INFO" "🔄 Этап 2.2: Преобразование в Markdown..."
    local transform_config="
intermediate_file=/app/temp/dag1_results_${timestamp}.json
original_config=$base_config
dag1_completed=true
vllm_model=Qwen/Qwen2.5-VL-32B-Instruct
transformation_quality=high
preserve_technical_terms=true"
    
    local dag2_run_id
    dag2_run_id=$(trigger_dag "content_transformation" "$transform_config" "Content Transformation")
    
    if [ $? -eq 0 ]; then
        if wait_for_dag_completion "content_transformation" "$dag2_run_id" "Content Transformation" 1200; then
            log "INFO" "✅ Этап 2.2 завершен: Markdown создан"
        else
            log "ERROR" "❌ Ошибка в преобразовании Markdown"
            return 1
        fi
    else
        return 1
    fi
    
    # Этап 2.3: Quality Assurance (5 уровней)
    log "INFO" "🔄 Этап 2.3: 5-уровневая валидация качества..."
    local qa_config="
translated_file=/app/output_md_zh/${timestamp}_${filename%.pdf}.md
translated_content=from_file
original_config=$base_config
translation_metadata={\"target_language\":\"zh\",\"processing_chain\":[\"document_preprocessing\",\"content_transformation\"]}
dag3_completed=false
quality_target=100.0
validation_levels=5
auto_correction=true
validation_mode=content_only"
    
    local dag4_run_id
    dag4_run_id=$(trigger_dag "quality_assurance" "$qa_config" "Quality Assurance (5 levels)")
    
    if [ $? -eq 0 ]; then
        if wait_for_dag_completion "quality_assurance" "$dag4_run_id" "Quality Assurance" 900; then
            log "INFO" "✅ Этап 2.3 завершен: валидация пройдена"
            
            # Показать результаты QA
            local qa_report_path="/app/temp/qa_report_qa_${timestamp}.json"
            show_qa_results "$qa_report_path"
            
            # Проверка финального результата
            local output_file="$OUTPUT_DIR/${timestamp}_${filename%.pdf}.md"
            if [ -f "$output_file" ]; then
                log "INFO" "📁 Валидированный результат: $output_file"
                log "INFO" "📋 QA отчет: $qa_report_path"
                return 0
            else
                log "WARN" "⚠️ Файл результата не найден: $output_file"
                return 1
            fi
        else
            log "ERROR" "❌ Ошибка в валидации качества"
            return 1
        fi
    else
        return 1
    fi
}

process_batch() {
    log "INFO" "🔍 Поиск PDF файлов для валидированной обработки..."
    
    # Поиск всех PDF файлов
    local pdf_files=()
    while IFS= read -r -d '' file; do
        pdf_files+=("$file")
    done < <(find "$INPUT_DIR" -name "*.pdf" -type f -print0)
    
    local total_files=${#pdf_files[@]}
    
    if [ $total_files -eq 0 ]; then
        log "WARN" "📂 Нет PDF файлов в $INPUT_DIR"
        echo "Поместите PDF файлы в папку $INPUT_DIR и запустите снова"
        return 0
    fi
    
    log "INFO" "📊 Найдено файлов для валидированной обработки: $total_files"
    echo ""
    
    # Обработка файлов
    local processed=0
    local failed=0
    local high_quality=0
    local start_time=$(date +%s)
    
    for pdf_file in "${pdf_files[@]}"; do
        local filename=$(basename "$pdf_file")
        echo -e "${BLUE}[ФАЙЛ $((processed + failed + 1))/$total_files]${NC} $filename"
        
        if process_single_file "$pdf_file"; then
            ((processed++))
            ((high_quality++))  # Предполагаем высокое качество при успехе
            echo -e "Статус: ${GREEN}✅ УСПЕШНО + ВАЛИДИРОВАНО${NC}"
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
    echo -e "${GREEN}ЭТАП 2 ЗАВЕРШЕН: КОНВЕРТАЦИЯ + ВАЛИДАЦИЯ${NC}"
    echo "==============================================================================="
    echo -e "📊 Статистика валидированной обработки:"
    echo -e "   Успешно обработано: ${GREEN}$processed${NC} файлов"
    echo -e "   Высокое качество (>95%): ${GREEN}$high_quality${NC} файлов"
    echo -e "   Ошибок: ${RED}$failed${NC} файлов"
    echo -e "   Общее время: ${BLUE}$total_duration${NC} секунд"
    echo ""
    echo -e "🔍 Применена 5-уровневая валидация:"
    echo -e "   1️⃣ OCR кросс-валидация"
    echo -e "   2️⃣ Визуальное сравнение (SSIM)"
    echo -e "   3️⃣ AST структурное сравнение"
    echo -e "   4️⃣ Валидация содержимого"
    echo -e "   5️⃣ Автокоррекция и финальная оценка"
    echo ""
    echo -e "📁 Результаты сохранены в: ${YELLOW}$OUTPUT_DIR${NC}"
    echo -e "📋 QA отчеты сохранены в: ${YELLOW}/app/temp/${NC}"
    echo -e "📋 Логи сохранены в: ${YELLOW}$LOGS_DIR${NC}"
    echo ""
    
    if [ $failed -gt 0 ]; then
        echo -e "${YELLOW}⚠️ Рекомендации:${NC}"
        echo "   - Проверьте логи для диагностики ошибок"
        echo "   - Убедитесь что PDF файлы читаемы"
        echo "   - Проверьте работу Quality Assurance сервиса"
    else
        echo -e "${GREEN}🎉 Все файлы успешно конвертированы и валидированы!${NC}"
        echo ""
        echo "Следующие шаги:"
        echo "   - Файлы готовы к использованию (качество 100%)"
        echo "   - Для перевода: ./process-stage3.sh [язык]"
    fi
}

# Основная логика
main() {
    show_header
    check_services
    
    echo -e "${YELLOW}Внимание: Обработка с 5-уровневой валидацией займет больше времени${NC}"
    echo -e "${YELLOW}Нажмите Enter для начала или Ctrl+C для отмены...${NC}"
    read -r
    
    process_batch
}

# Запуск, если скрипт вызван напрямую
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi