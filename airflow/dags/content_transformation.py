# DAG 2: Content Transformation (ИСПРАВЛЕНО)
# Преобразование извлеченного контента в высококачественный Markdown
# Использование vLLM с Qwen2.5-VL-32B-Instruct

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import json
import re
import os
from typing import Dict, List, Any, Optional

# Импорт кастомных утилит
from shared_utils import (
    SharedUtils,
    NotificationUtils,
    ConfigUtils
)

# Конфигурация DAG
DEFAULT_ARGS = {
    'owner': 'pdf-converter',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Создание DAG
dag = DAG(
    'content_transformation',
    default_args=DEFAULT_ARGS,
    description='DAG 2: Преобразование контента в высококачественный Markdown',
    schedule_interval=None,  # Запускается после DAG 1
    max_active_runs=2,
    catchup=False,
    tags=['pdf-converter', 'content-transformation', 'markdown', 'vllm']
)

# =============================================================================
# vLLM CLIENT ДЛЯ CONTENT TRANSFORMATION
# =============================================================================
class VLLMContentTransformer:
    """Клиент для преобразования контента через vLLM"""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
    
    def get_transformation_prompt(self, content_type: str) -> str:
        """Системный промпт для трансформации контента"""
        base_prompt = """Ты эксперт по преобразованию извлеченного из PDF контента в высококачественный Markdown.

ОСНОВНАЯ ЗАДАЧА: Создать идеальный Markdown документ с сохранением всей структуры и технических деталей.

КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ:
1. Сохрани ВСЮ структуру документа (заголовки, списки, таблицы)
2. Создай корректную иерархию заголовков (# ## ### ####)
3. Оформи технические команды IPMI/BMC/Redfish в блоках кода ```
4. Сохрани ВСЕ числовые данные и спецификации БЕЗ изменений
5. Создай корректные Markdown таблицы с правильным выравниванием
6. НЕ добавляй пояснения или комментарии - только чистый Markdown
7. Сохрани все технические термины и аббревиатуры

ФОРМАТИРОВАНИЕ:
- Заголовки: используй # ## ### #### по иерархии
- Таблицы: обязательно с заголовками и разделителями |---|---|
- Код/команды: ```bash или ```json для блоков
- Списки: - для маркированных, 1. для нумерованных
- Выделение: **жирный** для важных терминов

ТЕХНИЧЕСКИЕ ТЕРМИНЫ СОХРАНЯТЬ:
- IPMI команды: chassis, power, mc, sensor, sel, sdr, fru
- BMC команды: reset, info, watchdog, lan, user
- Redfish API: Systems, Chassis, Managers, UpdateService
- Аппаратные спецификации: CPU, RAM, SSD, NIC, GPU
- Модели устройств и серийные номера"""

        if content_type == "complex_table":
            return base_prompt + "\n\nОСОБОЕ ВНИМАНИЕ: Этот контент содержит сложные таблицы. Создай идеальные Markdown таблицы с правильным количеством столбцов и строк."
        elif content_type == "technical_specs":
            return base_prompt + "\n\nОСОБОЕ ВНИМАНИЕ: Этот контент содержит технические спецификации. Сохрани ВСЕ числовые значения, единицы измерения и модели устройств."
        elif content_type == "commands":
            return base_prompt + "\n\nОСОБОЕ ВНИМАНИЕ: Этот контент содержит технические команды. Оформи их в блоки кода с соответствующей подсветкой синтаксиса."
        
        return base_prompt
    
    def transform_content_chunk(self, text_chunk: str, structure_info: Dict = None, content_type: str = "mixed") -> Optional[str]:
        """Преобразование фрагмента контента в Markdown"""
        try:
            system_prompt = self.get_transformation_prompt(content_type)
            
            # Подготовка пользовательского промпта
            user_prompt = f"""Преобразуй этот извлеченный из PDF текст в идеальный Markdown:

ИСХОДНЫЙ КОНТЕНТ:
{text_chunk}"""

            # Добавление информации о структуре, если есть
            if structure_info:
                user_prompt += f"""

СТРУКТУРНАЯ ИНФОРМАЦИЯ:
{json.dumps(structure_info, ensure_ascii=False, indent=2)}"""
            
            # Запрос к vLLM
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 4096,
                "top_p": 0.9,
                "stream": False
            }
            
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=600
            )
            
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                markdown_content = result["choices"][0]["message"]["content"]
                return self.postprocess_markdown(markdown_content)
                
        except Exception as e:
            print(f"❌ Ошибка трансформации контента: {e}")
            return None
    
    def postprocess_markdown(self, markdown: str) -> str:
        """Постобработка сгенерированного Markdown"""
        if not markdown:
            return ""
        
        # Очистка от лишних элементов
        cleaned = markdown.strip()
        
        # Удаление возможных вводных фраз
        intro_patterns = [
            r'^[Вв]от Markdown версия[^:]*:?\s*',
            r'^[Вв]от преобразованный контент[^:]*:?\s*',
            r'^[Mm]arkdown версия[^:]*:?\s*',
            r'^[Hh]ere is[^:]*:?\s*'
        ]
        
        for pattern in intro_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # Исправление заголовков (убедимся что есть пробел после #)
        cleaned = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', cleaned, flags=re.MULTILINE)
        
        # Исправление таблиц - добавление разделителей если отсутствуют
        lines = cleaned.split('\n')
        corrected_lines = []
        in_table = False
        
        for i, line in enumerate(lines):
            if '|' in line and line.count('|') >= 2:
                if not in_table:
                    # Начало таблицы - проверяем следующую строку
                    if i + 1 < len(lines) and not re.match(r'^\|[\s\-\|:]+\|', lines[i + 1]):
                        corrected_lines.append(line)
                        # Добавляем разделитель
                        cols = line.count('|') - 1
                        separator = '|' + '---|' * cols
                        corrected_lines.append(separator)
                        in_table = True
                        continue
                    in_table = True
                corrected_lines.append(line)
            else:
                if in_table and line.strip() == '':
                    in_table = False
                corrected_lines.append(line)
        
        cleaned = '\n'.join(corrected_lines)
        
        # Финальная очистка множественных пустых строк
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        
        return cleaned.strip()

# =============================================================================
# ФУНКЦИИ DAG
# =============================================================================

def load_extraction_data(**context):
    """Загрузка данных извлечения от DAG 1"""
    dag_run_conf = context['dag_run'].conf
    
    # Получение промежуточного файла от DAG 1
    intermediate_file = dag_run_conf.get('intermediate_file')
    
    if not intermediate_file or not os.path.exists(intermediate_file):
        raise ValueError(f"Промежуточный файл от DAG 1 не найден: {intermediate_file}")
    
    # Чтение данных извлечения
    with open(intermediate_file, 'r', encoding='utf-8') as f:
        extraction_data = json.load(f)
    
    if not extraction_data or 'extracted_content' not in extraction_data:
        raise ValueError("Данные извлечения некорректны или отсутствуют")
    
    # Подготовка данных для трансформации
    transformation_session = {
        'session_id': f"transform_{int(datetime.now().timestamp())}",
        'extraction_data': extraction_data['extracted_content'],
        'analysis_data': extraction_data.get('analysis', {}),
        'source_file': extraction_data.get('source_file'),
        'original_config': dag_run_conf.get('original_config', {}),
        'vllm_model': 'Qwen/Qwen2.5-VL-32B-Instruct',  # ИСПРАВЛЕННАЯ МОДЕЛЬ
        'transformation_start_time': datetime.now().isoformat()
    }
    
    extracted_text = transformation_session['extraction_data'].get('extracted_text', '')
    tables_count = len(transformation_session['extraction_data'].get('tables', []))
    images_count = len(transformation_session['extraction_data'].get('images', []))
    
    print(f"📄 Данные загружены для трансформации:")
    print(f"   Текст: {len(extracted_text)} символов")
    print(f"   Таблицы: {tables_count}")
    print(f"   Изображения: {images_count}")
    
    return transformation_session

def analyze_content_structure(**context):
    """Анализ структуры контента для оптимальной трансформации"""
    transformation_session = context['task_instance'].xcom_pull(task_ids='load_extraction_data')
    
    extraction_data = transformation_session['extraction_data']
    extracted_text = extraction_data.get('extracted_text', '')
    document_structure = extraction_data.get('document_structure', {})
    tables = extraction_data.get('tables', [])
    
    # Анализ типов контента
    content_analysis = {
        'total_length': len(extracted_text),
        'has_structure': bool(document_structure),
        'tables_count': len(tables),
        'images_count': len(extraction_data.get('images', [])),
        'content_types': []
    }
    
    # Разделение текста на смысловые блоки
    text_blocks = split_into_blocks(extracted_text)
    
    analyzed_blocks = []
    for i, block in enumerate(text_blocks):
        block_type = classify_content_block(block)
        block_info = {
            'block_id': i,
            'content': block,
            'type': block_type,
            'length': len(block),
            'complexity': calculate_complexity(block)
        }
        analyzed_blocks.append(block_info)
        content_analysis['content_types'].append(block_type)
    
    # Группировка блоков для эффективной обработки
    processing_groups = group_blocks_for_processing(analyzed_blocks)
    
    structure_analysis = {
        'content_analysis': content_analysis,
        'analyzed_blocks': analyzed_blocks,
        'processing_groups': processing_groups,
        'recommended_approach': determine_processing_approach(content_analysis)
    }
    
    print(f"📊 Анализ структуры завершен:")
    print(f"   Блоков контента: {len(analyzed_blocks)}")
    print(f"   Групп обработки: {len(processing_groups)}")
    print(f"   Рекомендуемый подход: {structure_analysis['recommended_approach']}")
    
    return structure_analysis

def split_into_blocks(text: str) -> List[str]:
    """Разделение текста на смысловые блоки"""
    # Разделение по параграфам и заголовкам
    blocks = []
    current_block = []
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Новый блок при встрече заголовка или после пустой строки
        if (line.startswith('#') or 
            (not line and current_block) or
            len('\n'.join(current_block)) > 1000):
            
            if current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
        
        if line:
            current_block.append(line)
    
    if current_block:
        blocks.append('\n'.join(current_block))
    
    return [block for block in blocks if block.strip()]

def classify_content_block(block: str) -> str:
    """Классификация типа блока контента"""
    if not block.strip():
        return 'empty'
    
    # Заголовки
    if any(line.strip().isupper() and len(line.strip()) < 100 for line in block.split('\n')[:2]):
        return 'header'
    
    # Таблицы (по характерным признакам)
    table_indicators = ['|', 'колонка', 'столбец', 'строка', 'таблица']
    if any(indicator in block.lower() for indicator in table_indicators):
        return 'table'
    
    # Технические команды
    tech_indicators = ['ipmitool', 'redfish', 'bmc', '0x', 'chassis', 'power']
    if any(indicator in block.lower() for indicator in tech_indicators):
        return 'commands'
    
    # Технические спецификации
    if re.search(r'\d+\s*(GB|MHz|GHz|W|TB|MB|Gbps|RPM)', block):
        return 'technical_specs'
    
    # Списки
    list_patterns = [r'^\s*[-•]\s+', r'^\s*\d+[\.\)]\s+', r'^\s*[a-zA-Z]\)']
    if any(re.search(pattern, block, re.MULTILINE) for pattern in list_patterns):
        return 'list'
    
    # Обычный текст
    return 'text'

def calculate_complexity(block: str) -> str:
    """Расчет сложности блока"""
    length = len(block)
    
    if length > 1000:
        return 'high'
    elif length > 300:
        return 'medium'
    else:
        return 'low'

def group_blocks_for_processing(blocks: List[Dict]) -> List[List[Dict]]:
    """Группировка блоков для эффективной обработки"""
    groups = []
    current_group = []
    current_group_length = 0
    max_group_length = 2000
    
    for block in blocks:
        block_length = block['length']
        
        # Сложные блоки обрабатываются отдельно
        if block['complexity'] == 'high' or block['type'] in ['table', 'commands']:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_group_length = 0
            
            groups.append([block])
            continue
        
        # Проверка превышения лимита группы
        if current_group_length + block_length > max_group_length and current_group:
            groups.append(current_group)
            current_group = [block]
            current_group_length = block_length
        else:
            current_group.append(block)
            current_group_length += block_length
    
    if current_group:
        groups.append(current_group)
    
    return groups

def determine_processing_approach(analysis: Dict) -> str:
    """Определение оптимального подхода к обработке"""
    if analysis['tables_count'] > 5:
        return 'table_focused'
    elif 'commands' in analysis['content_types']:
        return 'technical_focused'
    elif analysis['total_length'] > 10000:
        return 'chunked_processing'
    else:
        return 'standard'

def transform_content_blocks(**context):
    """Трансформация блоков контента в Markdown"""
    transformation_session = context['task_instance'].xcom_pull(task_ids='load_extraction_data')
    structure_analysis = context['task_instance'].xcom_pull(task_ids='analyze_content_structure')
    
    # Инициализация vLLM трансформера
    transformer = VLLMContentTransformer(
        base_url=ConfigUtils.get_service_config()['vllm'],
        model=transformation_session['vllm_model']
    )
    
    processing_groups = structure_analysis['processing_groups']
    extraction_data = transformation_session['extraction_data']
    
    print(f"🔄 Начало трансформации {len(processing_groups)} групп контента")
    
    transformed_blocks = []
    processing_stats = {
        'groups_processed': 0,
        'blocks_processed': 0,
        'transformation_errors': 0
    }
    
    for group_idx, group in enumerate(processing_groups):
        print(f"📦 Обработка группы {group_idx + 1}/{len(processing_groups)}")
        
        # Объединение блоков группы
        group_content = '\n\n'.join([block['content'] for block in group])
        group_types = [block['type'] for block in group]
        
        # Определение типа группы
        if 'table' in group_types:
            content_type = 'complex_table'
        elif 'commands' in group_types:
            content_type = 'commands'
        elif 'technical_specs' in group_types:
            content_type = 'technical_specs'
        else:
            content_type = 'mixed'
        
        # Трансформация группы
        transformed_content = transformer.transform_content_chunk(
            text_chunk=group_content,
            structure_info=extraction_data.get('document_structure', {}),
            content_type=content_type
        )
        
        if transformed_content:
            transformed_blocks.append(transformed_content)
            processing_stats['groups_processed'] += 1
            processing_stats['blocks_processed'] += len(group)
        else:
            print(f"⚠️ Не удалось трансформировать группу {group_idx + 1}")
            processing_stats['transformation_errors'] += 1
            # Добавляем исходный контент как fallback
            transformed_blocks.append(f"```\n{group_content}\n```")
    
    # Объединение всех трансформированных блоков
    final_markdown = '\n\n'.join(transformed_blocks)
    
    # Добавление изображений если есть
    images = extraction_data.get('images', [])
    if images:
        final_markdown += "\n\n## Изображения\n\n"
        for i, image_path in enumerate(images):
            final_markdown += f"![Image {i+1}]({image_path})\n\n"
    
    transformation_results = {
        'markdown_content': final_markdown,
        'processing_stats': processing_stats,
        'content_length': len(final_markdown),
        'transformation_quality': calculate_transformation_quality(final_markdown, extraction_data)
    }
    
    print(f"✅ Трансформация завершена:")
    print(f"   Markdown: {len(final_markdown)} символов")
    print(f"   Качество: {transformation_results['transformation_quality']}%")
    
    return transformation_results

def calculate_transformation_quality(markdown: str, extraction_data: Dict) -> float:
    """Расчет качества трансформации"""
    quality_score = 100.0
    
    # Проверка наличия заголовков
    headers = len(re.findall(r'^#{1,6}\s+', markdown, re.MULTILINE))
    if headers == 0:
        quality_score -= 20
    
    # Проверка таблиц
    original_tables = len(extraction_data.get('tables', []))
    markdown_tables = len(re.findall(r'^\|.*\|', markdown, re.MULTILINE))
    if original_tables > 0 and markdown_tables < original_tables * 0.8:
        quality_score -= 15
    
    # Проверка структуры
    if not re.search(r'^#\s+', markdown, re.MULTILINE):  # Нет главного заголовка
        quality_score -= 10
    
    # Проверка кодовых блоков
    code_blocks = len(re.findall(r'```', markdown)) // 2
    if 'ipmitool' in extraction_data.get('extracted_text', '').lower() and code_blocks == 0:
        quality_score -= 10
    
    return max(0, quality_score)

def save_markdown_result(**context):
    """Сохранение результата трансформации"""
    transformation_session = context['task_instance'].xcom_pull(task_ids='load_extraction_data')
    transformation_results = context['task_instance'].xcom_pull(task_ids='transform_content_blocks')
    
    original_config = transformation_session['original_config']
    timestamp = original_config.get('timestamp', int(datetime.now().timestamp()))
    filename = original_config.get('filename', 'unknown.pdf')
    
    # Сохранение в папку для китайского языка (исходный Markdown)
    output_path = SharedUtils.prepare_output_path(filename, 'zh', timestamp)
    
    # Сохранение Markdown файла
    SharedUtils.save_final_result(
        content=transformation_results['markdown_content'],
        output_path=output_path,
        metadata={
            'source_file': transformation_session['source_file'],
            'transformation_model': transformation_session['vllm_model'],
            'processing_stats': transformation_results['processing_stats'],
            'transformation_quality': transformation_results['transformation_quality'],
            'content_length': transformation_results['content_length'],
            'processing_time': (datetime.now() - datetime.fromisoformat(transformation_session['transformation_start_time'])).total_seconds()
        }
    )
    
    # Конфигурация для следующего DAG
    dag3_config = {
        'markdown_file': output_path,
        'markdown_content': transformation_results['markdown_content'],
        'original_config': original_config,
        'dag2_completed': True,
        'transformation_quality': transformation_results['transformation_quality']
    }
    
    print(f"💾 Markdown сохранен: {output_path}")
    return dag3_config

# =============================================================================
# ОПРЕДЕЛЕНИЕ ЗАДАЧ DAG
# =============================================================================

# Задача 1: Загрузка данных извлечения
load_data = PythonOperator(
    task_id='load_extraction_data',
    python_callable=load_extraction_data,
    dag=dag
)

# Задача 2: Анализ структуры контента
analyze_structure = PythonOperator(
    task_id='analyze_content_structure',
    python_callable=analyze_content_structure,
    dag=dag
)

# Задача 3: Трансформация блоков контента
transform_blocks = PythonOperator(
    task_id='transform_content_blocks',
    python_callable=transform_content_blocks,
    dag=dag
)

# Задача 4: Сохранение результата
save_result = PythonOperator(
    task_id='save_markdown_result',
    python_callable=save_markdown_result,
    dag=dag
)

def notify_completion(**context):
    """Уведомление о завершении трансформации"""
    dag3_config = context['task_instance'].xcom_pull(task_ids='save_markdown_result')
    transformation_session = context['task_instance'].xcom_pull(task_ids='load_extraction_data')
    
    quality = dag3_config['transformation_quality']
    
    message = f"""
    ✅ DAG 2 (Content Transformation) успешно завершен
    
    Модель: {transformation_session['vllm_model']}
    Качество трансформации: {quality}%
    Markdown файл: {dag3_config['markdown_file']}
    
    Следующий этап: Translation Pipeline (DAG 3)
    """
    
    print(message)
    NotificationUtils.send_success_notification(context, dag3_config)

# Задача 5: Уведомление
notify_task = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    dag=dag
)

# Определение зависимостей задач
load_data >> analyze_structure >> transform_blocks >> save_result >> notify_task

# Настройка обработки ошибок
def handle_failure(context):
    """Обработка ошибок в DAG"""
    NotificationUtils.send_failure_notification(context, context.get('exception'))

# Применение обработчика ошибок ко всем задачам
for task in dag.tasks:
    task.on_failure_callback = handle_failure