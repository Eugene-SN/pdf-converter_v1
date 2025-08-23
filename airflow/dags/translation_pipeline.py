# DAG 3: Translation Pipeline (ИСПРАВЛЕНО)
# Высококачественный перевод с сохранением структуры на основе ollama-translator
# Интеграция с vLLM API вместо Ollama

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import requests
import json
import re
import time
import hashlib
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
    'translation_pipeline',
    default_args=DEFAULT_ARGS,
    description='DAG 3: Высококачественный перевод с сохранением структуры',
    schedule_interval=None,  # Запускается после DAG 2
    max_active_runs=2,
    catchup=False,
    tags=['pdf-converter', 'translation', 'vllm', 'technical-terms']
)

# =============================================================================
# ТЕХНИЧЕСКИЕ ТЕРМИНЫ (НЕ ПЕРЕВОДИТЬ!)
# =============================================================================
TECHNICAL_TERMINOLOGY = {
    # Brand names (КРИТИЧЕСКИ ВАЖНО!)
    "问天": "WenTian",
    "联想问天": "Lenovo WenTian",
    "天擎": "ThinkSystem",
    "AnyBay": "AnyBay",
    
    # Processors
    "至强": "Xeon",
    "可扩展处理器": "Scalable Processors",
    "英特尔": "Intel",
    "处理器": "Processor",
    "内核": "Core",
    "线程": "Thread",
    "睿频": "Turbo Boost",
    
    # Memory and storage
    "内存": "Memory",
    "存储": "Storage",
    "硬盘": "Drive",
    "固态硬盘": "SSD",
    "机械硬盘": "HDD",
    "热插拔": "Hot-swap",
    "冗余": "Redundancy",
    "背板": "Backplane",
    "托架": "Tray",
    
    # Network
    "以太网": "Ethernet",
    "光纤": "Fiber",
    "带宽": "Bandwidth",
    "延迟": "Latency",
    "网卡": "Network Adapter",
    
    # Form factors
    "英寸": "inch",
    "机架": "Rack",
    "插槽": "Slot",
    "转接卡": "Riser Card",
    
    # Power
    "电源": "Power Supply",
    "铂金": "Platinum",
    "钛金": "Titanium",
    "CRPS": "CRPS"
}

TECHNICAL_TERMINOLOGY_RU = {
    "问天": "WenTian",
    "联想问天": "Lenovo WenTian",
    "至强": "Xeon",
    "可扩展处理器": "Scalable процессоры",
    "英特尔": "Intel",
    "处理器": "процессор",
    "内核": "ядро",
    "线程": "поток",
    "内存": "память",
    "存储": "хранилище",
    "硬盘": "диск",
    "固态硬盘": "SSD",
    "机械硬盘": "HDD",
    "热插拔": "горячая замена",
    "冗余": "резервирование",
    "托架": "лоток",
    "以太网": "Ethernet",
    "带宽": "пропускная способность",
    "延迟": "задержка",
    "机架": "стойка",
    "插槽": "слот",
    "电源": "блок питания"
}

# =============================================================================
# vLLM API КЛИЕНТ (АДАПТИРОВАННЫЙ ИЗ OLLAMA-TRANSLATOR)
# =============================================================================
class VLLMTranslationClient:
    """Клиент для работы с vLLM API для перевода"""
    
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.session = requests.Session()
        self.translation_cache = {}
        
    def get_system_prompt(self, source_lang: str, target_lang: str) -> str:
        """Системный промпт с отключением thinking режима"""
        return f"""Вы - эксперт по переводу технической документации серверного оборудования.

ОСНОВНАЯ ЗАДАЧА: Переводите технические тексты с {source_lang} на {target_lang} язык.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:
1. ТОЛЬКО готовый перевод, БЕЗ пояснений и комментариев
2. НИКОГДА не используйте фразы "Вот перевод", "Here is translation"
3. НЕ добавляйте размышления в тегах <думаю>, или любых других
4. Сохраняйте ВСЕ числа, коды, форматирование ТОЧНО
5. НЕ используйте thinking режим - отвечайте сразу и кратко

БРЕНДЫ (НЕ МЕНЯТЬ!):
- "问天" ВСЕГДА → "WenTian"
- "联想问天" ВСЕГДА → "Lenovo WenTian"
- Сохраняйте: Intel, Xeon, PCIe, DDR5, NVMe, SAS, SATA

КОМАНДЫ НЕ ПЕРЕВОДИТЬ:
- IPMI: chassis, power, mc, sensor, sel, sdr, fru
- API: get, set, list, status, activate, deactivate
- Hex коды: 0x30, 0x02
- Аббревиатуры: GPU, CPU, SSD, HDD, RAID, BMC

ФОРМАТИРОВАНИЕ:
- Сохраняйте структуру Markdown таблиц ТОЧНО
- НЕ изменяйте количество столбцов
- Сохраняйте HTML теги и отступы

Выводите ТОЛЬКО {target_lang} перевод без дополнительного текста!"""

    def get_user_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Пользовательский промпт"""
        return f"""Переведите этот {source_lang} технический текст на {target_lang} язык:

{text}"""

    def make_translation_request(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Выполнение запроса перевода к vLLM"""
        try:
            # Проверка кэша
            cache_key = self.get_cache_key(text, source_lang, target_lang)
            if cache_key in self.translation_cache:
                print(f"📦 Получен перевод из кэша")
                return self.translation_cache[cache_key]

            # Подготовка промптов
            system_prompt = self.get_system_prompt(source_lang, target_lang)
            user_prompt = self.get_user_prompt(text, source_lang, target_lang)
            
            # Запрос к vLLM API
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
                translated = result["choices"][0]["message"]["content"]
                
                # Постобработка
                cleaned = self.postprocess_translation(translated, target_lang)
                
                # Сохранение в кэш
                self.translation_cache[cache_key] = cleaned
                
                return cleaned
                
        except Exception as e:
            print(f"❌ Ошибка перевода: {e}")
            return None
    
    def postprocess_translation(self, response: str, target_lang: str) -> str:
        """Постобработка перевода с очисткой thinking режима"""
        if not response:
            return ""
        
        cleaned = response.strip()
        
        # Удаление thinking тегов и размышлений
        thinking_patterns = [
            r'<думаю>.*?</думаю>',
            r'<thinking>.*?</thinking>',
            r'[Хх]орошо[,\s]*мне нужно[^.]*?\.',
            r'[Сс]начала посмотр[^.]*?\.',
            r'Let me[^.]*?\.',
            r'First[,\s]*I[^.]*?\.',
            r'[Вв]от перевод[^:]*:?\s*',
            r'Here is[^:]*:?\s*',
            r'[Нн]иже представлен[^:]*:?\s*'
        ]
        
        for pattern in thinking_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Исправление технических терминов
        cleaned = self.fix_technical_terms(cleaned, target_lang)
        
        # Финальная очистка
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def fix_technical_terms(self, text: str, target_lang: str) -> str:
        """Исправление технических терминов"""
        # Критически важно: НЕ МЕНЯТЬ WenTian на ThinkSystem!
        brand_fixes = {
            r'\b[Qq]itian\b': 'WenTian',
            r'\bSkyland\b': 'WenTian',
            r'\bSkyStorage\b': 'WenTian'
        }
        
        for wrong_pattern, correct in brand_fixes.items():
            if re.search(wrong_pattern, text):
                text = re.sub(wrong_pattern, correct, text)
        
        # Выбор словаря терминов в зависимости от языка
        if target_lang == "ru":
            terminology = TECHNICAL_TERMINOLOGY_RU
        else:
            terminology = TECHNICAL_TERMINOLOGY
        
        # Применение технических терминов
        for chinese_term, translation in terminology.items():
            if chinese_term in text:
                text = text.replace(chinese_term, translation)
        
        return text
    
    def get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Создание ключа для кэша"""
        content = f"{source_lang}-{target_lang}-{text}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

# =============================================================================
# ФУНКЦИИ DAG
# =============================================================================

def initialize_translation(**context):
    """Инициализация процесса перевода"""
    dag_run_conf = context['dag_run'].conf
    
    # Получение конфигурации от предыдущего DAG
    markdown_file = dag_run_conf.get('markdown_file')
    original_config = dag_run_conf.get('original_config', {})
    
    target_language = original_config.get('target_language', 'ru')
    
    # Чтение Markdown контента
    if markdown_file and SharedUtils.validate_input_file(markdown_file.replace('.md', '.pdf')):
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        except:
            # Fallback: читаем из конфигурации
            markdown_content = dag_run_conf.get('markdown_content', '')
    else:
        markdown_content = dag_run_conf.get('markdown_content', '')
    
    if not markdown_content:
        raise ValueError("Нет Markdown контента для перевода")
    
    # Инициализация сессии перевода
    translation_session = {
        'session_id': f"translation_{int(datetime.now().timestamp())}",
        'source_language': 'zh-CN',
        'target_language': target_language,
        'markdown_content': markdown_content,
        'original_config': original_config,
        'translation_model': 'Qwen/Qwen3-30B-A3B-Instruct-2507',  # ИСПРАВЛЕННАЯ МОДЕЛЬ
        'preserve_technical_terms': True,
        'batch_processing': True,
        'lines_total': len(markdown_content.split('\n')),
        'processing_start_time': datetime.now().isoformat()
    }
    
    print(f"🌐 Инициализация перевода: {target_language}")
    print(f"📊 Строк для перевода: {translation_session['lines_total']}")
    
    return translation_session

def batch_translate_content(**context):
    """Пакетный перевод контента с оптимизацией"""
    translation_session = context['task_instance'].xcom_pull(task_ids='initialize_translation')
    
    markdown_content = translation_session['markdown_content']
    target_language = translation_session['target_language']
    
    # Инициализация vLLM клиента
    vllm_client = VLLMTranslationClient(
        base_url=ConfigUtils.get_service_config()['vllm'],
        model=translation_session['translation_model']
    )
    
    print(f"🚀 Начало пакетного перевода на {target_language}")
    
    # Разделение на строки для батчинга
    lines = markdown_content.split('\n')
    
    # Интеллектуальное батчирование по типу контента
    batches = create_smart_batches(lines)
    
    translated_lines = []
    batch_count = 0
    
    for batch in batches:
        batch_count += 1
        batch_content = '\n'.join(batch)
        
        if not batch_content.strip():
            translated_lines.extend(batch)
            continue
        
        print(f"📦 Обработка батча {batch_count}/{len(batches)}")
        
        # Перевод батча
        translated_batch = vllm_client.make_translation_request(
            batch_content, 'китайский', get_language_name(target_language)
        )
        
        if translated_batch:
            # Разбиение обратно на строки
            batch_lines = translated_batch.split('\n')
            if len(batch_lines) == len(batch):
                translated_lines.extend(batch_lines)
            else:
                # Если разбиение не удалось, переводим построчно
                for line in batch:
                    if line.strip():
                        translated_line = vllm_client.make_translation_request(
                            line, 'китайский', get_language_name(target_language)
                        )
                        translated_lines.append(translated_line or line)
                    else:
                        translated_lines.append(line)
        else:
            # Если перевод батча не удался, оставляем оригинал
            translated_lines.extend(batch)
        
        # Небольшая задержка между батчами
        time.sleep(0.2)
    
    # Объединение результата
    translated_content = '\n'.join(translated_lines)
    
    # Валидация качества перевода
    quality_score = validate_translation_quality(markdown_content, translated_content, target_language)
    
    translation_results = {
        'translated_content': translated_content,
        'source_lines': len(lines),
        'batches_processed': len(batches),
        'quality_score': quality_score,
        'translation_stats': {
            'input_length': len(markdown_content),
            'output_length': len(translated_content),
            'chinese_remaining': len(re.findall(r'[\u4e00-\u9fff]', translated_content))
        }
    }
    
    print(f"✅ Перевод завершен. Качество: {quality_score}%")
    return translation_results

def create_smart_batches(lines: List[str]) -> List[List[str]]:
    """Интеллектуальное создание батчей по типу контента"""
    batches = []
    current_batch = []
    max_batch_size = 6
    
    for line in lines:
        content_type = analyze_line_complexity(line)
        optimal_batch_size = get_optimal_batch_size(content_type)
        
        if len(current_batch) >= optimal_batch_size and current_batch:
            batches.append(current_batch)
            current_batch = [line]
        else:
            current_batch.append(line)
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def analyze_line_complexity(line: str) -> str:
    """Анализ сложности строки для оптимального батчинга"""
    if not line.strip():
        return 'empty'
    
    if line.strip().startswith('#'):
        return 'header'
    
    if '|' in line and line.count('|') > 2:
        return 'table'
    
    if any(re.search(pattern, line) for pattern in [r'\d+\s*(GB|MB|GHz|MHz|W|TB)', r'ipmitool', r'0x[0-9a-f]+']):
        return 'technical_specs'
    
    if len(line) > 100:
        return 'long_text'
    
    return 'text'

def get_optimal_batch_size(content_type: str) -> int:
    """Оптимальный размер батча для типа контента"""
    batch_sizes = {
        'empty': 20,
        'header': 4,
        'table': 2,
        'technical_specs': 3,
        'long_text': 1,
        'text': 6
    }
    return batch_sizes.get(content_type, 4)

def get_language_name(lang_code: str) -> str:
    """Получение полного имени языка"""
    languages = {
        'ru': 'русский',
        'en': 'английский',
        'zh': 'китайский'
    }
    return languages.get(lang_code, lang_code)

def validate_translation_quality(original: str, translated: str, target_lang: str) -> float:
    """Валидация качества перевода (адаптировано из ollama-translator)"""
    quality_score = 100.0
    
    # 1. Проверка китайских символов
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', translated))
    if chinese_count > 0:
        quality_score -= min(50, chinese_count * 3)
    
    # 2. Проверка размышлений и thinking
    thinking_patterns = [
        r'[Хх]орошо[,\s]*мне', r'[Сс]начала посмотр', r'Let me', r'First I',
        r'[Вв]от перевод', r'Here is', r'[Нн]иже представлен',
        r'<думаю>', r'</думаю>', r'<thinking>', r'</thinking>'
    ]
    
    thinking_count = sum(len(re.findall(pattern, translated)) for pattern in thinking_patterns)
    if thinking_count > 0:
        quality_score -= min(30, thinking_count * 10)
    
    # 3. Проверка правильности перевода брендов
    if "问天" in original and "WenTian" not in translated:
        quality_score -= 20
    
    # 4. Проверка структуры таблиц
    orig_table_rows = len(re.findall(r'^\|.*\|$', original, re.MULTILINE))
    trans_table_rows = len(re.findall(r'^\|.*\|$', translated, re.MULTILINE))
    
    if orig_table_rows > 0:
        table_preservation = trans_table_rows / orig_table_rows
        if table_preservation < 0.9:
            quality_score -= 15
    
    # 5. Проверка размера
    size_ratio = len(translated) / max(len(original), 1)
    if size_ratio > 2.5 or size_ratio < 0.4:
        quality_score -= 10
    
    return max(0, quality_score)

def intelligent_fix_remaining(**context):
    """Интеллектуальное исправление оставшихся китайских символов"""
    translation_results = context['task_instance'].xcom_pull(task_ids='batch_translate_content')
    translation_session = context['task_instance'].xcom_pull(task_ids='initialize_translation')
    
    translated_content = translation_results['translated_content']
    chinese_remaining = translation_results['translation_stats']['chinese_remaining']
    
    if chinese_remaining == 0:
        print("✅ Китайских символов не обнаружено, исправление не требуется")
        return translation_results
    
    print(f"🔧 Обнаружено {chinese_remaining} китайских символов, запускаем исправление...")
    
    # Инициализация клиента для исправления
    vllm_client = VLLMTranslationClient(
        base_url=ConfigUtils.get_service_config()['vllm'],
        model=translation_session['translation_model']
    )
    
    # Поиск китайских фрагментов
    chinese_fragments = []
    for match in re.finditer(r'[\u4e00-\u9fff]+', translated_content):
        fragment = match.group()
        if len(fragment) >= 2:  # Минимальная длина фрагмента
            chinese_fragments.append((fragment, match.span()))
    
    # Исправление по одному фрагменту
    current_text = translated_content
    fixes_count = 0
    target_language = translation_session['target_language']
    
    for fragment, span in chinese_fragments[:10]:  # Максимум 10 исправлений
        translated_fragment = vllm_client.make_translation_request(
            fragment, 'китайский', get_language_name(target_language)
        )
        
        if translated_fragment and fragment != translated_fragment and not re.search(r'[\u4e00-\u9fff]', translated_fragment):
            current_text = current_text.replace(fragment, translated_fragment, 1)
            fixes_count += 1
            print(f"✅ Исправлен #{fixes_count}: {fragment} → {translated_fragment[:20]}...")
    
    # Обновление результатов
    final_chinese_count = len(re.findall(r'[\u4e00-\u9fff]', current_text))
    improvement = chinese_remaining - final_chinese_count
    
    print(f"📈 Результат исправления: {chinese_remaining} → {final_chinese_count} (исправлено: {improvement})")
    
    translation_results['translated_content'] = current_text
    translation_results['translation_stats']['chinese_remaining'] = final_chinese_count
    translation_results['fixes_applied'] = fixes_count
    
    return translation_results

def save_translation_result(**context):
    """Сохранение результата перевода"""
    translation_session = context['task_instance'].xcom_pull(task_ids='initialize_translation')
    translation_results = context['task_instance'].xcom_pull(task_ids='intelligent_fix_remaining')
    
    original_config = translation_session['original_config']
    target_language = translation_session['target_language']
    timestamp = original_config.get('timestamp', int(datetime.now().timestamp()))
    filename = original_config.get('filename', 'unknown.pdf')
    
    # Определение пути сохранения
    output_path = SharedUtils.prepare_output_path(filename, target_language, timestamp)
    
    # Сохранение переведенного контента
    SharedUtils.save_final_result(
        content=translation_results['translated_content'],
        output_path=output_path,
        metadata={
            'source_file': original_config.get('input_file'),
            'target_language': target_language,
            'translation_model': translation_session['translation_model'],
            'translation_stats': translation_results['translation_stats'],
            'quality_score': translation_results['quality_score'],
            'processing_time': (datetime.now() - datetime.fromisoformat(translation_session['processing_start_time'])).total_seconds(),
            'fixes_applied': translation_results.get('fixes_applied', 0)
        }
    )
    
    # Результат для следующего DAG
    dag4_config = {
        'translated_file': output_path,
        'translated_content': translation_results['translated_content'],
        'translation_metadata': {
            'target_language': target_language,
            'quality_score': translation_results['quality_score'],
            'source_lines': translation_results['source_lines'],
            'translation_stats': translation_results['translation_stats']
        },
        'original_config': original_config,
        'dag3_completed': True
    }
    
    print(f"💾 Результат перевода сохранен: {output_path}")
    return dag4_config

# =============================================================================
# ОПРЕДЕЛЕНИЕ ЗАДАЧ DAG
# =============================================================================

# Задача 1: Инициализация перевода
init_translation = PythonOperator(
    task_id='initialize_translation',
    python_callable=initialize_translation,
    dag=dag
)

# Задача 2: Пакетный перевод контента
batch_translate = PythonOperator(
    task_id='batch_translate_content',
    python_callable=batch_translate_content,
    dag=dag
)

# Задача 3: Исправление остатков
fix_remaining = PythonOperator(
    task_id='intelligent_fix_remaining',
    python_callable=intelligent_fix_remaining,
    dag=dag
)

# Задача 4: Сохранение результата
save_result = PythonOperator(
    task_id='save_translation_result',
    python_callable=save_translation_result,
    dag=dag
)

def notify_completion(**context):
    """Уведомление о завершении перевода"""
    dag4_config = context['task_instance'].xcom_pull(task_ids='save_translation_result')
    translation_session = context['task_instance'].xcom_pull(task_ids='initialize_translation')
    
    target_language = translation_session['target_language']
    quality_score = dag4_config['translation_metadata']['quality_score']
    
    message = f"""
    ✅ DAG 3 (Translation Pipeline) успешно завершен
    
    Язык перевода: {target_language}
    Качество: {quality_score:.1f}%
    Модель: {translation_session['translation_model']}
    Файл: {dag4_config['translated_file']}
    
    Следующий этап: Quality Assurance (DAG 4)
    """
    
    print(message)
    NotificationUtils.send_success_notification(context, dag4_config)

# Задача 5: Уведомление
notify_task = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_completion,
    dag=dag
)

# Определение зависимостей задач
init_translation >> batch_translate >> fix_remaining >> save_result >> notify_task

# Настройка обработки ошибок
def handle_failure(context):
    """Обработка ошибок в DAG"""
    NotificationUtils.send_failure_notification(context, context.get('exception'))

# Применение обработчика ошибок ко всем задачам
for task in dag.tasks:
    task.on_failure_callback = handle_failure