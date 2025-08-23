#!/bin/bash

# =============================================================================
# PDF CONVERTER PIPELINE v2.0 - ЭТАП 3: ПОЛНЫЙ ПАЙПЛАЙН
# Запускает все 4 DAG через orchestrator_dag: конвертация + перевод + валидация
# Результат: PDF → Markdown на целевом языке с полной валидацией (цель: 100%)
# =============================================================================

set -euo pipefail

# Конфигурация
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/input_pdf"
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

# Поддерживаемые языки
declare -A LANGUAGES=(
    ["ru"]="русский"
    ["en"]="английский"
    ["zh"]="китайский"
    ["original"]="оригинальный"
)

# Создание директорий
mkdir -p "$INPUT_DIR" "$LOGS_DIR"

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[$timestamp]${NC} ${YELLOW}[$level]${NC} $message" | tee -a "$LOGS_DIR/stage3_$(date +%Y%m%d_%H%M%S).log"
}

show_header() {
    echo -e "${BLUE}"
    echo "==============================================================================="
    echo "  PDF CONVERTER PIPELINE v2.0 - ЭТАП 3: ПОЛНЫЙ ПАЙПЛАЙН"
    echo "==============================================================================="
    echo -e "${NC}"
    echo "🎯 Цель: PDF → Переведенный Markdown с полной валидацией"
    echo "📂 Входная папка: $INPUT_DIR"
    echo ""
    echo "🔄 Выполняемые этапы:"
    echo "   1️⃣ DAG 1: Document Preprocessing (Извлечение контента)"
    echo "   2️⃣ DAG 2: Content Transformation (Markdown Qwen2.5-VL-32B)"
    echo "   3️⃣ DAG 3: Translation Pipeline (Перевод Qwen3-30B-A3B)"
    echo "   4️⃣ DAG 4: Quality Assurance (5-уровневая валидация)"
    echo ""
    echo "🤖 Используемые модели:"
    echo "   • Content: Qwen/Qwen2.5-VL-32B-Instruct"
    echo "   • Translation: Qwen/Qwen3-30B-A3B-Instruct-2507"
    echo ""
}

show_usage() {
    echo "Использование: $0 [ЯЗЫК]"
    echo ""
    echo "Доступные языки:"
    for lang_code in "${!LANGUAGES[@]}"; do
        echo "  $lang_code - ${LANGUAGES[$lang_code]}"
    done
    echo ""
    echo "Примеры:"
    echo "  $0 ru     # Перевод на русский"
    echo "  $0 en     # Перевод на английский"
    echo "  $0        # Интерактивный выбор языка"
    echo ""
}

select_target_language() {
    local target_lang="$1"
    
    # Если язык передан как аргумент
    if [ -n "$target_lang" ]; then
        if [ -v "LANGUAGES[$target_lang]" ]; then
            echo "$target_lang"
            return 0
        else
            log "ERROR" "Неподдерживаемый язык: $target_lang"
            show_usage
            exit 1
        fi
    fi
    
    # Интерактивный выбор языка
    echo -e "${YELLOW}Выберите целевой язык для перевода:${NC}"
    echo ""
    local i=1
    local lang_array=()
    
    for lang_code in "${!LANGUAGES[@]}"; do
        lang_array+=("$lang_code")
        echo "  $i) $lang_code - ${LANGUAGES[$lang_code]}"
        ((i++))
    done
    
    echo ""
    echo -n "Введите номер (1-${#LANGUAGES[@]}): "
    read -r choice
    
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#LANGUAGES[@]} ]; then
        local selected_lang="${lang_array[$((choice-1))]}"
        log "INFO" "Выбран язык: $selected_lang - ${LANGUAGES[$selected_lang]}"
        echo "$selected_lang"
    else
        log "ERROR" "Неверный выбор: $choice"
        exit 1
    fi
}

get_output_dir() {
    local target_lang="$1"
    echo "${SCRIPT_DIR}/output_md_${target_lang}"
}

check_services() {
    log "INFO" "Проверка готовности всех сервисов полного пайплайна..."
    
    local services=(
        "$AIRFLOW_BASE_URL/health:Airflow UI"
        "${DOCUMENT_PROCESSOR_URL:-http://localhost:8001}/health:Document Processor"
        "${VLLM_SERVER_URL:-http://localhost:8000}/health:vLLM Server"
        "${QUALITY_ASSURANCE_URL:-http://localhost:8002}/health:Quality Assurance"
        "${TRANSLATOR_URL:-http://localhost:8003}/health:Translator"
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
                log "INFO" "Убедитесь что система запущена: ./start-system.sh"
                exit 1
            fi
        fi
        
        log "INFO" "✅ $name готов"
    done
    
    log "INFO" "✅ Все сервисы для полного пайплайна готовы"
}

trigger_orchestrator() {
    local pdf_file="$1"
    local filename="$2"
    local target_lang="$3"
    local timestamp="$4"
    
    log "INFO" "🚀 Запуск Master Orchestrator для полного пайплайна..."
    
    # Конфигурация для orchestrator_dag (все 4 DAG)
    local orchestrator_config=$(python3 -c "
import json
config = {
    'input_file': '$pdf_file',
    'filename': '$filename',
    'target_language': '$target_lang',
    'timestamp': $timestamp,
    'quality_level': 'high',
    'enable_ocr': True,
    'preserve_structure': True,
    'extract_tables': True,
    'extract_images': True,
    'batch_mode': False,
    'pipeline_version': '2.0_modular',
    'full_pipeline': True,
    'models': {
        'content_transformation': 'Qwen/Qwen2.5-VL-32B-Instruct',
        'translation': 'Qwen/Qwen3-30B-A3B-Instruct-2507'
    },
    'quality_target': 100.0,
    'validation_levels': 5,
    'auto_correction': True
}
print(json.dumps(config))
")
    
    # Запуск orchestrator_dag через API
    local response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        --user "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" \
        -H "Content-Type: application/json" \
        -d "{\"conf\": $orchestrator_config}" \
        "$AIRFLOW_BASE_URL/api/v1/dags/orchestrator_dag/dagRuns")
    
    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n -1)
    
    if [ "$http_code" -eq 200 ] || [ "$http_code" -eq 201 ]; then
        local dag_run_id=$(echo "$body" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('dag_run_id', 'unknown'))" 2>/dev/null || echo "unknown")
        log "INFO" "✅ Master Orchestrator запущен. Run ID: $dag_run_id"
        echo "$dag_run_id"
    else
        log "ERROR" "❌ Ошибка запуска Master Orchestrator: HTTP $http_code"
        log "ERROR" "Ответ: $body"
        return 1
    fi
}

monitor_orchestrator_progress() {
    local dag_run_id="$1"
    local filename="$2"
    local target_lang="$3"
    local timeout=3600  # 1 час для полного пайплайна
    
    log "INFO" "⏳ Мониторинг выполнения полного пайплайна (таймаут: ${timeout}s)..."
    
    local start_time=$(date +%s)
    local last_stage=""
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout ]; then
            log "ERROR" "❌ Таймаут выполнения полного пайплайна"
            return 1
        fi
        
        # Получение статуса orchestrator_dag
        local response=$(curl -s \
            --user "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" \
            "$AIRFLOW_BASE_URL/api/v1/dags/orchestrator_dag/dagRuns/$dag_run_id")
        
        local state=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('state', 'unknown'))
except:
    print('error')
" 2>/dev/null || echo "error")
        
        # Получение информации о текущих задачах
        local tasks_response=$(curl -s \
            --user "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" \
            "$AIRFLOW_BASE_URL/api/v1/dags/orchestrator_dag/dagRuns/$dag_run_id/taskInstances")
        
        local current_stage=$(echo "$tasks_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    task_instances = data.get('task_instances', [])
    
    stage_mapping = {
        'trigger_dag1_preprocessing': 'DAG 1: Document Preprocessing',
        'trigger_dag2_transformation': 'DAG 2: Content Transformation', 
        'trigger_dag3_translation': 'DAG 3: Translation Pipeline',
        'trigger_dag4_quality_assurance': 'DAG 4: Quality Assurance'
    }
    
    for task in task_instances:
        task_id = task.get('task_id', '')
        task_state = task.get('state', '')
        
        if task_state == 'running' and task_id in stage_mapping:
            print(stage_mapping[task_id])
            sys.exit(0)
    
    # Если нет запущенных задач, найдем последнюю успешную
    for task in reversed(task_instances):
        task_id = task.get('task_id', '')
        task_state = task.get('state', '')
        
        if task_state == 'success' and task_id in stage_mapping:
            print(f\"{stage_mapping[task_id]} ✅\")
            sys.exit(0)
    
    print('Инициализация...')
except:
    print('Проверка статуса...')
" 2>/dev/null || echo "Проверка статуса...")
        
        # Показать текущий этап если изменился
        if [ "$current_stage" != "$last_stage" ]; then
            if [ -n "$current_stage" ]; then
                log "INFO" "🔄 $current_stage"
                last_stage="$current_stage"
            fi
        fi
        
        case "$state" in
            "success")
                log "INFO" "✅ Полный пайплайн завершен успешно"
                return 0
                ;;
            "failed"|"upstream_failed")
                log "ERROR" "❌ Полный пайплайн завершен с ошибкой"
                return 1
                ;;
            "running")
                # Показать прогресс
                local minutes=$((elapsed / 60))
                printf "\r${YELLOW}[ВЫПОЛНЯЕТСЯ $minutes мин]${NC} $current_stage   "
                sleep 10
                ;;
            *)
                sleep 5
                ;;
        esac
    done
}

show_final_results() {
    local filename="$1"
    local target_lang="$2"
    local timestamp="$3"
    local processing_time="$4"
    
    local output_dir=$(get_output_dir "$target_lang")
    local output_file="$output_dir/${timestamp}_${filename%.pdf}.md"
    local qa_report="/app/temp/qa_report_qa_${timestamp}.json"
    
    echo ""
    echo "==============================================================================="
    echo -e "${GREEN}🎉 ПОЛНЫЙ ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!${NC}"
    echo "==============================================================================="
    echo ""
    echo -e "📄 Исходный файл: ${BLUE}$filename${NC}"
    echo -e "🌐 Целевой язык: ${BLUE}${LANGUAGES[$target_lang]}${NC}"
    echo -e "⏱️ Время обработки: ${BLUE}$processing_time${NC}"
    echo ""
    echo -e "🤖 Использованные модели:"
    echo -e "   • Content Transformation: ${YELLOW}Qwen/Qwen2.5-VL-32B-Instruct${NC}"
    echo -e "   • Translation: ${YELLOW}Qwen/Qwen3-30B-A3B-Instruct-2507${NC}"
    echo ""
    echo -e "🔄 Выполненные этапы:"
    echo -e "   ✅ DAG 1: Document Preprocessing (Docling + OCR)"
    echo -e "   ✅ DAG 2: Content Transformation (vLLM Markdown)"
    echo -e "   ✅ DAG 3: Translation Pipeline (vLLM Translation)"
    echo -e "   ✅ DAG 4: Quality Assurance (5-уровневая валидация)"
    echo ""
    
    # Показать QA результаты если доступны
    if [ -f "$qa_report" ]; then
        local qa_summary=$(python3 -c "
import json
try:
    with open('$qa_report', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    quality_summary = data.get('quality_summary', {})
    overall_score = quality_summary.get('overall_score', 'N/A')
    quality_grade = quality_summary.get('quality_grade', 'N/A')
    target_achieved = quality_summary.get('target_achieved', False)
    
    print(f\"Score: {overall_score}\")
    print(f\"Grade: {quality_grade}\")
    print(f\"Target: {target_achieved}\")
except:
    print(\"Score: N/A\")
    print(\"Grade: N/A\")  
    print(\"Target: N/A\")
")
        
        local score=$(echo "$qa_summary" | grep "Score:" | cut -d':' -f2 | xargs)
        local grade=$(echo "$qa_summary" | grep "Grade:" | cut -d':' -f2 | xargs)
        local target=$(echo "$qa_summary" | grep "Target:" | cut -d':' -f2 | xargs)
        
        echo -e "🎯 Результаты валидации качества:"
        echo -e "   • Общий балл: ${GREEN}$score%${NC}"
        echo -e "   • Оценка качества: ${BLUE}$grade${NC}"
        
        if [ "$target" = "True" ]; then
            echo -e "   • Цель 100%: ${GREEN}✅ ДОСТИГНУТА${NC}"
        else
            echo -e "   • Цель 100%: ${YELLOW}⚠️ НЕ ДОСТИГНУТА${NC}"
        fi
        echo ""
    fi
    
    echo -e "📁 Результаты сохранены:"
    echo -e "   • Переведенный файл: ${GREEN}$output_file${NC}"
    echo -e "   • QA отчет: ${YELLOW}$qa_report${NC}"
    echo -e "   • Логи: ${YELLOW}$LOGS_DIR${NC}"
    echo ""
    echo -e "${GREEN}🚀 СИСТЕМА v2.0 ГОТОВА К ПРОДАКШЕНУ!${NC}"
}

process_single_file() {
    local pdf_file="$1"
    local target_lang="$2"
    local filename=$(basename "$pdf_file")
    local timestamp=$(date +%s)
    local start_time=$(date +%s)
    
    log "INFO" "📄 Запуск полного пайплайна для: $filename"
    log "INFO" "🌐 Целевой язык: ${LANGUAGES[$target_lang]}"
    
    # Создание выходной директории
    local output_dir=$(get_output_dir "$target_lang")
    mkdir -p "$output_dir"
    
    # Запуск orchestrator_dag
    local orchestrator_run_id
    orchestrator_run_id=$(trigger_orchestrator "$pdf_file" "$filename" "$target_lang" "$timestamp")
    
    if [ $? -eq 0 ]; then
        if monitor_orchestrator_progress "$orchestrator_run_id" "$filename" "$target_lang"; then
            local end_time=$(date +%s)
            local processing_duration=$((end_time - start_time))
            local processing_time="${processing_duration} секунд"
            
            if [ $processing_duration -gt 60 ]; then
                local minutes=$((processing_duration / 60))
                local seconds=$((processing_duration % 60))
                processing_time="${minutes} мин ${seconds} сек"
            fi
            
            show_final_results "$filename" "$target_lang" "$timestamp" "$processing_time"
            return 0
        else
            log "ERROR" "❌ Ошибка в выполнении полного пайплайна"
            return 1
        fi
    else
        log "ERROR" "❌ Не удалось запустить orchestrator"
        return 1
    fi
}

process_batch() {
    local target_lang="$1"
    
    log "INFO" "🔍 Поиск PDF файлов для полного пайплайна..."
    
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
    
    log "INFO" "📊 Найдено файлов для полного пайплайна: $total_files"
    log "INFO" "🌐 Целевой язык: ${LANGUAGES[$target_lang]}"
    echo ""
    
    # Обработка файлов
    local processed=0
    local failed=0
    local total_start_time=$(date +%s)
    
    for pdf_file in "${pdf_files[@]}"; do
        local filename=$(basename "$pdf_file")
        echo -e "${BLUE}[ФАЙЛ $((processed + failed + 1))/$total_files]${NC} $filename → ${LANGUAGES[$target_lang]}"
        
        if process_single_file "$pdf_file" "$target_lang"; then
            ((processed++))
            echo -e "Статус: ${GREEN}✅ ПОЛНЫЙ УСПЕХ${NC}"
        else
            ((failed++))
            echo -e "Статус: ${RED}❌ ОШИБКА${NC}"
        fi
        
        echo ""
    done
    
    # Финальная статистика
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    echo "==============================================================================="
    echo -e "${GREEN}ПОЛНЫЙ ПАЙПЛАЙН ЗАВЕРШЕН${NC}"
    echo "==============================================================================="
    echo -e "📊 Итоговая статистика:"
    echo -e "   Успешно обработано: ${GREEN}$processed${NC} файлов"
    echo -e "   Ошибок: ${RED}$failed${NC} файлов"
    echo -e "   Целевой язык: ${BLUE}${LANGUAGES[$target_lang]}${NC}"
    echo -e "   Общее время: ${BLUE}$total_duration${NC} секунд"
    echo ""
    
    local output_dir=$(get_output_dir "$target_lang")
    echo -e "📁 Все результаты в: ${YELLOW}$output_dir${NC}"
    echo -e "📋 Логи в: ${YELLOW}$LOGS_DIR${NC}"
    echo ""
    
    if [ $failed -gt 0 ]; then
        echo -e "${YELLOW}⚠️ Рекомендации по ошибкам:${NC}"
        echo "   - Проверьте логи Airflow в веб-интерфейсе"
        echo "   - Убедитесь в стабильности vLLM сервиса"
        echo "   - Проверьте качество входных PDF файлов"
    else
        echo -e "${GREEN}🎉 ВСЕ ФАЙЛЫ УСПЕШНО ОБРАБОТАНЫ!${NC}"
        echo -e "${GREEN}🚀 МОДУЛЬНАЯ АРХИТЕКТУРА v2.0 РАБОТАЕТ ИДЕАЛЬНО!${NC}"
    fi
}

# Основная логика
main() {
    local target_lang="${1:-}"
    
    show_header
    
    # Выбор языка
    target_lang=$(select_target_language "$target_lang")
    local output_dir=$(get_output_dir "$target_lang")
    
    echo -e "📁 Результаты будут сохранены в: ${YELLOW}$output_dir${NC}"
    echo ""
    
    check_services
    
    echo -e "${YELLOW}⚠️ ВНИМАНИЕ: Полный пайплайн может занять продолжительное время${NC}"
    echo -e "${YELLOW}Каждый файл проходит все 4 этапа с валидацией качества${NC}"
    echo -e "${YELLOW}Нажмите Enter для начала или Ctrl+C для отмены...${NC}"
    read -r
    
    process_batch "$target_lang"
}

# Показать справку, если передан -h или --help
if [[ "${1:-}" =~ ^(-h|--help)$ ]]; then
    show_usage
    exit 0
fi

# Запуск, если скрипт вызван напрямую
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi