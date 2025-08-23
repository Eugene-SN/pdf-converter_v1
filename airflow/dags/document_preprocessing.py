# DAG 1: Document Preprocessing
# Извлечение контента из PDF с использованием Docling + OCR + Tabula

import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.decorators import apply_defaults

# Импорт наших кастомных операторов
from shared_utils import (
    DocumentProcessorOperator,
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
    'document_preprocessing',
    default_args=DEFAULT_ARGS,
    description='DAG 1: Извлечение контента из PDF документов',
    schedule_interval=None,  # Запускается по требованию
    max_active_runs=3,
    catchup=False,
    tags=['pdf-converter', 'preprocessing', 'docling', 'ocr']
)

def validate_input_task(**context):
    """Валидация входного PDF файла"""
    input_file = context['dag_run'].conf.get('input_file')
    
    if not input_file:
        raise ValueError("Не указан входной файл")
    
    if not SharedUtils.validate_input_file(input_file):
        raise ValueError(f"Недопустимый файл: {input_file}")
    
    # Получение метаданных файла
    import os
    file_stats = os.stat(input_file)
    
    metadata = {
        'file_path': input_file,
        'file_size_bytes': file_stats.st_size,
        'file_size_mb': round(file_stats.st_size / (1024*1024), 2),
        'file_name': os.path.basename(input_file),
        'validation_timestamp': datetime.now().isoformat()
    }
    
    print(f"✅ Валидация входного файла пройдена: {metadata}")
    return metadata

def prepare_processing_config(**context):
    """Подготовка конфигурации для обработки"""
    dag_conf = context['dag_run'].conf
    
    # Настройки обработки по умолчанию + переданные параметры
    processing_config = {
        # OCR настройки
        'use_ocr': dag_conf.get('use_ocr', False),
        'ocr_languages': dag_conf.get('ocr_languages', 'chi_sim,chi_tra,eng,rus'),
        'ocr_engine_primary': 'paddleocr',
        'ocr_engine_secondary': 'tesseract',
        
        # Docling настройки
        'docling_device': dag_conf.get('docling_device', 'cuda'),
        'layout_analysis': True,
        'structure_detection': True,
        
        # Извлечение элементов
        'extract_tables': dag_conf.get('extract_tables', True),
        'extract_images': dag_conf.get('extract_images', True),
        'extract_formulas': dag_conf.get('extract_formulas', True),
        'extract_code_blocks': True,
        
        # Tabula-Py для таблиц
        'tabula_enabled': True,
        'tabula_pages': 'all',
        'tabula_lattice': True,
        'tabula_stream': True,
        
        # Качество извлечения
        'quality_level': dag_conf.get('quality_level', 'high'),
        'preserve_formatting': True,
        'preserve_structure': dag_conf.get('preserve_structure', True),
        
        # Оптимизации для китайских документов
        'chinese_optimization': True,
        'traditional_simplified_conversion': True,
        
        # Технические настройки
        'max_processing_time': 1800,  # 30 минут
        'memory_limit': '4G'
    }
    
    print(f"📋 Конфигурация обработки: {processing_config}")
    return processing_config

# Задача 1: Валидация входного файла
validate_input = PythonOperator(
    task_id='validate_input_file',
    python_callable=validate_input_task,
    dag=dag
)

# Задача 2: Подготовка конфигурации
prepare_config = PythonOperator(
    task_id='prepare_processing_config',
    python_callable=prepare_processing_config,
    dag=dag
)

# Задача 3: Извлечение контента через Document Processor
extract_content = DocumentProcessorOperator(
    task_id='extract_content',
    input_file_path="{{ dag_run.conf['input_file'] }}",
    processing_options="{{ json.dumps(task_instance.xcom_pull(task_ids='prepare_processing_config')) }}",
    timeout=1800, # 30 минут    
    dag=dag
)

def analyze_extraction_results(**context):
    """Анализ результатов извлечения контента"""
    extraction_data = context['task_instance'].xcom_pull(task_ids='extract_content')
    
    if not extraction_data:
        raise ValueError("Не получены данные от Document Processor")
    
    # Анализ качества извлечения
    text_length = len(extraction_data.get('extracted_text', ''))
    tables_count = len(extraction_data.get('tables', []))
    images_count = len(extraction_data.get('images', []))
    
    analysis = {
        'extraction_quality': {
            'text_extracted': text_length > 0,
            'text_length': text_length,
            'tables_found': tables_count,
            'images_found': images_count,
            'has_structure': bool(extraction_data.get('document_structure'))
        },
        'processing_stats': extraction_data.get('processing_stats', {}),
        'readiness_for_next_dag': True
    }
    
    # Проверки качества
    if text_length < 10:
        print("⚠️ ПРЕДУПРЕЖДЕНИЕ: Извлечено мало текста")
        analysis['readiness_for_next_dag'] = False
    
    if tables_count == 0 and 'table' in extraction_data.get('extracted_text', '').lower():
        print("⚠️ ПРЕДУПРЕЖДЕНИЕ: Возможно пропущены таблицы")
    
    # Подготовка данных для следующего DAG
    dag_2_input = {
        'extracted_content': extraction_data,
        'analysis': analysis,
        'source_file': context['dag_run'].conf.get('input_file'),
        'processing_timestamp': datetime.now().isoformat()
    }
    
    print(f"📊 Анализ извлечения: {analysis}")
    return dag_2_input

# Задача 4: Анализ результатов извлечения
analyze_results = PythonOperator(
    task_id='analyze_extraction_results',
    python_callable=analyze_extraction_results,
    dag=dag
)

def save_intermediate_results(**context):
    """Сохранение промежуточных результатов"""
    dag_2_input = context['task_instance'].xcom_pull(task_ids='analyze_extraction_results')
    timestamp = context['dag_run'].conf.get('timestamp', int(datetime.now().timestamp()))
    filename = context['dag_run'].conf.get('filename', 'unknown.pdf')
    
    # Сохранение данных для следующего DAG
    intermediate_path = f"/tmp/dag1_results_{timestamp}.json"
    
    import json
    import os
    
    os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
    
    with open(intermediate_path, 'w', encoding='utf-8') as f:
        json.dump(dag_2_input, f, ensure_ascii=False, indent=2)
    
    # Установка прав доступа с обработкой ошибки
    try:
        os.chown(intermediate_path, 1000, 1000)
    except PermissionError:
        print("⚠️ Не удалось изменить владельца файла, продолжаем без изменения прав")
        pass
    
    # Подготовка конфигурации для DAG 2
    next_dag_config = {
        'intermediate_file': intermediate_path,
        'original_config': context['dag_run'].conf,
        'dag1_completed': True,
        'ready_for_dag2': dag_2_input['analysis']['readiness_for_next_dag']
    }
    
    print(f"💾 Промежуточные результаты сохранены: {intermediate_path}")
    return next_dag_config

# Задача 5: Сохранение промежуточных результатов
save_results = PythonOperator(
    task_id='save_intermediate_results',
    python_callable=save_intermediate_results,
    dag=dag
)

def notify_completion(**context):
    """Уведомление о завершении DAG 1"""
    results = context['task_instance'].xcom_pull(task_ids='save_intermediate_results')
    dag_run_id = context['dag_run'].run_id
    
    if results and results.get('ready_for_dag2'):
        message = f"""
        ✅ DAG 1 (Document Preprocessing) успешно завершен
        
        Run ID: {dag_run_id}
        Файл: {context['dag_run'].conf.get('filename', 'unknown')}
        
        Результаты:
        - Текст извлечен: ✅
        - Промежуточные данные: {results.get('intermediate_file')}
        - Готовность для DAG 2: {'✅ Да' if results.get('ready_for_dag2') else '❌ Нет'}
        
        Следующий этап: Content Transformation (DAG 2)
        """
    else:
        message = f"""
        ❌ DAG 1 (Document Preprocessing) завершен с проблемами
        
        Run ID: {dag_run_id}
        Необходима проверка качества извлечения
        """
    
    print(message)
    
    # Отправка в систему мониторинга
    NotificationUtils.send_success_notification(context, results or {})

# Задача 6: Уведомление о завершении
notify_task = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    dag=dag
)

# Определение зависимостей задач
validate_input >> prepare_config >> extract_content >> analyze_results >> save_results >> notify_task

# Настройка обработки ошибок
def handle_failure(context):
    """Обработка ошибок в DAG"""
    NotificationUtils.send_failure_notification(context, context.get('exception'))

# Применение обработчика ошибок ко всем задачам
for task in dag.tasks:
    task.on_failure_callback = handle_failure

import json
dag.user_defined_macros = {'json': json}
