# Общие утилиты и операторы для модульных DAG - ИСПРАВЛЕНО для динамической подгрузки моделей
# PDF Converter Pipeline v2.0 с Dynamic vLLM Server
# ✅ ИСПРАВЛЕНА ПОТЕРЯ ДАННЫХ: правильное извлечение текста из Document Processor

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.utils.context import Context
import requests
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
import time

logger = logging.getLogger(__name__)

# =================================================================
# БАЗОВЫЙ КЛАСС ДЛЯ ВСЕХ ОПЕРАТОРОВ
# =================================================================

class PDFConverterBaseOperator(BaseOperator):
    """Базовый класс для всех операторов PDF конвейера"""

    @apply_defaults
    def __init__(
        self,
        service_endpoint: str,
        timeout: int = 300,
        retry_count: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.service_endpoint = service_endpoint
        self.timeout = timeout
        self.retry_count = retry_count

    def make_request(self, endpoint: str, method: str = 'POST', data: Dict = None, files: Dict = None) -> Dict:
        """Выполнение HTTP запроса к сервису с retry логикой"""
        url = f"{self.service_endpoint}{endpoint}"
        
        for attempt in range(self.retry_count):
            try:
                if method == 'POST':
                    if files:
                        response = requests.post(url, data=data, files=files, timeout=self.timeout)
                    else:
                        response = requests.post(url, json=data, timeout=self.timeout)
                else:
                    response = requests.get(url, timeout=self.timeout)
                    
                response.raise_for_status()
                return response.json()
                
            except Exception as e:
                logger.warning(f"Попытка {attempt + 1} неудачна: {str(e)}")
                if attempt == self.retry_count - 1:
                    raise
                    
        return {}

# =================================================================
# DYNAMIC vLLM OPERATOR - НОВЫЙ КЛАСС
# =================================================================

class DynamicVLLMOperator(PDFConverterBaseOperator):
    """
    Оператор для работы с Dynamic vLLM Server
    Автоматически выбирает и загружает нужную модель по типу задачи
    """

    @apply_defaults
    def __init__(
        self,
        task_type: str,  # 'content_transformation' или 'translation'
        system_prompt: str = None,
        user_prompt_template: str = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs
    ):
        super().__init__(
            service_endpoint=os.getenv('VLLM_SERVER_URL', 'http://vllm-server:8000'),
            timeout=1800,  # Увеличен таймаут для загрузки моделей
            **kwargs
        )
        
        self.task_type = task_type
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _wait_for_model_ready(self, max_wait_time: int = 600) -> bool:
        """Ожидание готовности модели к работе"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Проверяем статус через health endpoint
                response = requests.get(f"{self.service_endpoint}/health", timeout=10)
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data.get('status') == 'healthy':
                        logger.info("✅ Dynamic vLLM Server готов к работе")
                        return True
                    else:
                        logger.info(f"⏳ Ожидание готовности сервера: {health_data.get('status', 'unknown')}")
            except Exception as e:
                logger.debug(f"Проверка готовности не удалась: {e}")
                
            time.sleep(10)
        
        logger.error(f"❌ Тайм-аут ожидания готовности сервера ({max_wait_time}s)")
        return False

    def _get_current_model_info(self) -> Dict[str, Any]:
        """Получение информации о текущей модели"""
        try:
            response = requests.get(f"{self.service_endpoint}/v1/models/status", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Не удалось получить статус модели: {response.status_code}")
                return {}
        except Exception as e:
            logger.warning(f"Ошибка получения статуса модели: {e}")
            return {}

    def execute(self, context: Context) -> Dict[str, Any]:
        """Выполнение задачи через Dynamic vLLM API"""
        logger.info(f"🚀 Выполняем задачу типа: {self.task_type}")

        # Проверяем готовность сервера
        if not self._wait_for_model_ready():
            raise RuntimeError("Dynamic vLLM Server не готов к работе")

        # Получаем информацию о текущем состоянии
        model_status = self._get_current_model_info()
        logger.info(f"📊 Статус моделей: {model_status.get('manager_status', {})}")

        if self.task_type == 'content_transformation':
            return self._transform_content(context)
        elif self.task_type == 'translation':
            return self._translate_content(context)
        else:
            raise ValueError(f"Неподдерживаемый тип задачи: {self.task_type}")

    def _transform_content(self, context: Context) -> Dict[str, Any]:
        """Преобразование извлеченного контента в Markdown"""
        logger.info("🔄 Запуск Content Transformation")

        # Получение данных от предыдущего оператора
        upstream_data = context['task_instance'].xcom_pull(task_ids=None)
        if not upstream_data:
            raise ValueError("Нет данных от предыдущего этапа")

        # ✅ ИСПРАВЛЕНО: правильное извлечение текста из sections
        text_content = self._extract_full_text_from_sections(upstream_data)
        structure = upstream_data.get('document_structure', {})
        tables = upstream_data.get('tables', [])

        if not text_content:
            raise ValueError("Нет текстового контента для трансформации")

        # Формирование промпта для трансформации
        system_prompt = """Ты эксперт по преобразованию технических документов в высококачественный Markdown формат.

КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ:

1. Сохрани ВСЮ структуру документа (заголовки, подзаголовки, списки, таблицы)
2. Корректно оформи технические команды IPMI/BMC/Redfish в блоках кода ```
3. Сохрани все числовые данные и спецификации БЕЗ ИЗМЕНЕНИЙ
4. Создай правильную markdown-разметку для таблиц
5. Добавь соответствующие заголовки разных уровней (# ## ### ####)
6. НЕ используй thinking теги
7. Сохрани все специальные термины и аббревиатуры"""

        user_prompt = f"""Преобразуй извлеченный из PDF текст в идеальный Markdown формат.

ИСХОДНЫЙ ТЕКСТ:

{text_content[:8000]}  # Ограничиваем длину для модели

{"СТРУКТУРА ДОКУМЕНТА:" if structure else ""}
{json.dumps(structure, ensure_ascii=False, indent=2) if structure else ""}

{"ТАБЛИЦЫ:" if tables else ""}
{json.dumps(tables, ensure_ascii=False, indent=2) if tables else ""}

Создай чистый, структурированный Markdown без лишних комментариев."""

        # Подготовка запроса для Dynamic vLLM API
        request_data = {
            "model": "content-transformer",  # Алиас модели
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "task_type": "content_transformation",  # Указываем тип задачи для автовыбора модели
            "stream": False
        }

        # Выполнение запроса
        try:
            logger.info("📤 Отправка запроса в Dynamic vLLM Server")
            response = requests.post(
                f"{self.service_endpoint}/v1/chat/completions",
                json=request_data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()

            # Извлекаем результат
            if 'choices' in result and len(result['choices']) > 0:
                transformed_content = result['choices'][0]['message']['content']

                # Получаем метаинформацию о процессе
                pdf_meta = result.get('pdf_converter_meta', {})
                logger.info(f"✅ Трансформация завершена. Модель: {pdf_meta.get('model_key', 'unknown')}")
                logger.info(f"⏱️ Время обработки: {pdf_meta.get('processing_time_seconds', 0)}s")
                logger.info(f"🔄 Всего переключений моделей: {pdf_meta.get('model_swaps_total', 0)}")

                return {
                    'markdown_content': transformed_content,
                    'original_structure': structure,
                    'transformation_stats': {
                        'input_length': len(text_content),
                        'output_length': len(transformed_content),
                        'tables_processed': len(tables),
                        'model_used': pdf_meta.get('model_key', 'unknown'),
                        'processing_time': pdf_meta.get('processing_time_seconds', 0),
                        'vram_used_gb': pdf_meta.get('vram_usage_gb', 0)
                    }
                }
            else:
                raise ValueError("Пустой ответ от vLLM")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ошибка запроса к vLLM: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Ошибка обработки ответа: {e}")
            raise

    def _extract_full_text_from_sections(self, upstream_data: Dict[str, Any]) -> str:
        """✅ ИСПРАВЛЕНО: Правильное извлечение всего текста из sections Document Processor"""
        
        # Попробуем несколько путей извлечения данных
        full_text = ""
        
        try:
            # Путь 1: Прямое поле extracted_text
            if 'extracted_text' in upstream_data and upstream_data['extracted_text']:
                direct_text = upstream_data['extracted_text']
                if len(direct_text) > 50:  # Если есть достаточно текста
                    logger.info(f"✅ Использую прямое поле extracted_text: {len(direct_text)} символов")
                    return direct_text
            
            # Путь 2: Извлекаем из sections в metadata
            metadata = upstream_data.get('metadata', {})
            if metadata and 'structure_analysis' in metadata:
                structure = metadata['structure_analysis'].get('structure', {})
                
                # Извлекаем из paragraphs
                paragraphs = structure.get('paragraphs', [])
                for paragraph in paragraphs:
                    if 'text' in paragraph and paragraph['text']:
                        full_text += paragraph['text'] + "\n\n"
                
                # Извлекаем из headings
                headings = structure.get('headings', [])
                for heading in headings:
                    if 'text' in heading and heading['text']:
                        full_text += heading['text'] + "\n"
                
                if full_text and len(full_text) > 50:
                    logger.info(f"✅ Извлечено из structure_analysis: {len(full_text)} символов")
                    return full_text
            
            # Путь 3: Пробуем sections напрямую (если они есть в корне)
            sections = upstream_data.get('sections', [])
            if sections:
                for section in sections:
                    if isinstance(section, dict):
                        if 'title' in section and section['title']:
                            full_text += f"# {section['title']}\n\n"
                        if 'content' in section and section['content']:
                            full_text += section['content'] + "\n\n"
                
                if full_text and len(full_text) > 50:
                    logger.info(f"✅ Извлечено из sections: {len(full_text)} символов")
                    return full_text
            
            # Путь 4: Пробуем message поле (fallback)
            if 'message' in upstream_data and upstream_data['message']:
                message_text = upstream_data['message']
                if len(message_text) > 50:
                    logger.info(f"✅ Использую поле message: {len(message_text)} символов")
                    return message_text
            
            # Если ничего не найдено, логируем структуру для отладки
            logger.error("❌ Не удалось извлечь текст!")
            logger.error(f"🔍 Структура upstream_data: {list(upstream_data.keys())}")
            logger.error(f"🔍 Доступные поля: {upstream_data}")
            
            # Возвращаем что-то базовое
            return "Документ обработан, но текст не извлечен корректно"
            
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения текста: {e}")
            return f"Ошибка извлечения: {str(e)}"

    def _translate_content(self, context: Context) -> Dict[str, Any]:
        """Перевод содержимого с сохранением технических терминов"""
        logger.info("🌐 Запуск Translation Pipeline")

        # Получение данных от предыдущего этапа
        upstream_data = context['task_instance'].xcom_pull(task_ids=None)
        if not upstream_data:
            raise ValueError("Нет данных от предыдущего этапа")

        # Извлекаем Markdown контент
        markdown_content = upstream_data.get('markdown_content', '')
        if not markdown_content:
            raise ValueError("Нет Markdown контента для перевода")

        # Получаем целевой язык из конфигурации DAG
        target_language = context['dag_run'].conf.get('target_language', 'ru')
        
        if target_language == 'original':
            logger.info("🔄 Целевой язык 'original' - пропускаем перевод")
            return {
                'translated_content': markdown_content,
                'target_language': 'original',
                'translation_stats': {
                    'skipped': True,
                    'reason': 'original language requested'
                }
            }

        # Специальные промпты для разных языков
        language_prompts = {
            'ru': 'Переведи технический документ на русский язык',
            'en': 'Translate technical document to English', 
            'zh': '将技术文档翻译成中文'
        }

        system_prompt = f"""Ты эксперт по переводу технических документов на {target_language}.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА:

1. НЕ ПЕРЕВОДИ технические команды IPMI, BMC, Redfish
2. НЕ ПЕРЕВОДИ названия аппаратных устройств и модели
3. НЕ ПЕРЕВОДИ блоки кода в ``` и команды в `
4. СОХРАНИ всю Markdown разметку (заголовки, списки, таблицы)
5. СОХРАНИ все числовые значения и спецификации
6. ВАЖНО: "问天" ВСЕГДА переводи как "WenTian", НЕ как "ThinkSystem"
7. НЕ используй thinking теги
8. Переводи ТОЛЬКО описательный текст и комментарии"""

        user_prompt = f"""{language_prompts.get(target_language, language_prompts['en'])}

ИСХОДНЫЙ MARKDOWN:

{markdown_content[:8000]}  # Ограничиваем длину

Создай точный перевод, сохранив всю техническую информацию и структуру."""

        # Подготовка запроса для Dynamic vLLM API
        request_data = {
            "model": "translator",  # Алиас модели для перевода
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # Низкая температура для точности перевода
            "max_tokens": self.max_tokens,
            "task_type": "translation",  # Указываем тип задачи
            "stream": False
        }

        # Выполнение запроса
        try:
            logger.info("📤 Отправка запроса на перевод в Dynamic vLLM Server")
            response = requests.post(
                f"{self.service_endpoint}/v1/chat/completions",
                json=request_data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()

            # Извлекаем результат
            if 'choices' in result and len(result['choices']) > 0:
                translated_content = result['choices'][0]['message']['content']

                # Получаем метаинформацию
                pdf_meta = result.get('pdf_converter_meta', {})
                logger.info(f"✅ Перевод завершен. Модель: {pdf_meta.get('model_key', 'unknown')}")
                logger.info(f"⏱️ Время обработки: {pdf_meta.get('processing_time_seconds', 0)}s")

                return {
                    'translated_content': translated_content,
                    'target_language': target_language,
                    'original_content': markdown_content,
                    'translation_stats': {
                        'input_length': len(markdown_content),
                        'output_length': len(translated_content),
                        'model_used': pdf_meta.get('model_key', 'unknown'),
                        'processing_time': pdf_meta.get('processing_time_seconds', 0),
                        'vram_used_gb': pdf_meta.get('vram_usage_gb', 0)
                    }
                }
            else:
                raise ValueError("Пустой ответ от vLLM")

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ошибка запроса к vLLM: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Ошибка обработки перевода: {e}")
            raise

# =================================================================
# LEGACY OPERATOR (для обратной совместимости)
# =================================================================

class VLLMOperator(DynamicVLLMOperator):
    """
    Legacy оператор - алиас для DynamicVLLMOperator
    Сохраняется для обратной совместимости
    """
    def __init__(self, *args, **kwargs):
        logger.warning("VLLMOperator устарел. Используйте DynamicVLLMOperator")
        super().__init__(*args, **kwargs)

# =================================================================
# DOCUMENT PROCESSOR OPERATOR (без изменений)
# =================================================================

class DocumentProcessorOperator(PDFConverterBaseOperator):
    """Оператор для извлечения контента из PDF (DAG 1)"""
    
    template_fields = ('input_file_path', 'processing_options')

    @apply_defaults
    def __init__(
        self,
        input_file_path: str,
        processing_options: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(
            service_endpoint=os.getenv('DOCUMENT_PROCESSOR_URL', 'http://document-processor:8001'),
            **kwargs
        )
        
        self.input_file_path = input_file_path
        self.processing_options = processing_options or {}

    def execute(self, context: Context) -> Dict[str, Any]:
        """Извлечение контента из PDF документа"""
        import json
        
        logger.info(f"📄 Начинаем обработку документа: {self.input_file_path}")

        # Нормализация processing_options
        if isinstance(self.processing_options, str):
            try:
                self.processing_options = json.loads(self.processing_options)
                logger.info("✅ processing_options преобразованы из JSON в dict")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось распарсить processing_options: {e}")
                self.processing_options = {}

        if not isinstance(self.processing_options, dict):
            logger.warning("⚠️ processing_options не dict, используем значения по умолчанию")
            self.processing_options = {}

        # Подготовка файла для отправки
        with open(self.input_file_path, 'rb') as file:
            files = {'file': file}
            
            options_dict = {
                'use_ocr': self.processing_options.get('use_ocr', True),
                'extract_tables': self.processing_options.get('extract_tables', True),
                'extract_images': self.processing_options.get('extract_images', True),
                'analyze_structure': self.processing_options.get('analyze_structure', True),
                'docling_device': self.processing_options.get('docling_device', 'cuda')
            }
            
            data = {
                'options': json.dumps(options_dict)  # Сериализуем в JSON-строку
            }
            
            result = self.make_request('/convert', files=files, data=data)

        # ✅ ИСПРАВЛЕНО: правильная адаптация ответа Document Processor API v4.0
        
        # Извлекаем весь текст из всех sections
        full_extracted_text = ""
        sections_data = []
        
        # Парсим metadata для извлечения текста
        metadata = result.get('metadata', {})
        if metadata and 'structure_analysis' in metadata:
            structure = metadata['structure_analysis'].get('structure', {})
            
            # Извлекаем текст из paragraphs
            paragraphs = structure.get('paragraphs', [])
            for paragraph in paragraphs:
                if 'text' in paragraph and paragraph['text']:
                    full_extracted_text += paragraph['text'] + "\n\n"
            
            # Извлекаем текст из headings
            headings = structure.get('headings', [])
            for heading in headings:
                if 'text' in heading and heading['text']:
                    full_extracted_text += heading['text'] + "\n"
                    
                    # Создаем sections на основе headings
                    sections_data.append({
                        'title': heading['text'],
                        'level': heading.get('level', 1),
                        'page': heading.get('page', 1),
                        'content': '',  # Контент будет в paragraphs
                        'subsections': []
                    })

        # Если нет текста в structure, пробуем другие поля
        if not full_extracted_text:
            full_extracted_text = result.get('message', '')  # Fallback
            
        extracted_data = {
            'extracted_text': full_extracted_text,  # ✅ ИСПРАВЛЕНО: весь текст
            'sections': sections_data,  # ✅ ДОБАВЛЕНО: sections для следующих DAG
            'document_structure': {
                'pages': result.get('pages_count', 0),
                'sections': result.get('sections_count', 0),
                'tables': result.get('tables_count', 0),
                'images': result.get('images_count', 0),
                'formulas': result.get('formulas_count', 0)
            },
            'tables': [],  # Пока пустой - нужно уточнить где в API
            'images': [],  # Пока пустой - нужно уточнить где в API  
            'metadata': result.get('metadata', {}),
            'processing_stats': {
                'processing_time': result.get('processing_time', 0),
                'success': result.get('success', False),
                'document_id': result.get('document_id', ''),
                'output_files': result.get('output_files', [])
            }
        }

        logger.info(f"✅ Обработка завершена. Извлечено текста: {len(extracted_data['extracted_text'])} символов")
        
        return extracted_data

# =================================================================
# КАЧЕСТВО И ВАЛИДАЦИЯ (без изменений)  
# =================================================================

class QualityAssuranceOperator(PDFConverterBaseOperator):
    """Оператор для 5-уровневой валидации качества (DAG 4)"""

    @apply_defaults
    def __init__(
        self,
        validation_levels: List[str] = None,
        quality_target: float = 100.0,
        auto_correct: bool = True,
        **kwargs
    ):
        super().__init__(
            service_endpoint=os.getenv('QA_SERVICE_ENDPOINT', 'http://quality-assurance:8002'),
            **kwargs
        )
        
        self.validation_levels = validation_levels or ['ocr', 'visual', 'ast', 'content', 'correction']
        self.quality_target = quality_target
        self.auto_correct = auto_correct

    def execute(self, context: Context) -> Dict[str, Any]:
        """Выполнение 5-уровневой валидации"""
        logger.info("🔍 Запуск 5-уровневой системы валидации качества")

        # Получение данных от предыдущих этапов
        upstream_data = context['task_instance'].xcom_pull(task_ids=None)
        original_file = context['dag_run'].conf.get('input_file')

        # Подготовка данных для валидации
        validation_data = {
            'original_pdf': original_file,
            'processed_content': upstream_data,
            'validation_levels': self.validation_levels,
            'quality_target': self.quality_target,
            'auto_correct': self.auto_correct,
            'task_config': context['dag_run'].conf
        }

        # Запуск валидации
        result = self.make_request('/api/v1/validate', data=validation_data)

        validation_results = {
            'overall_score': result.get('overall_score', 0.0),
            'level_scores': result.get('level_scores', {}),
            'validation_report': result.get('report', {}),
            'corrections_applied': result.get('corrections', []),
            'final_content': result.get('final_content', ''),
            'quality_passed': result.get('overall_score', 0.0) >= self.quality_target
        }

        if not validation_results['quality_passed']:
            logger.warning(f"⚠️ Качество не достигнуто: {validation_results['overall_score']:.2f}% < {self.quality_target}%")
        else:
            logger.info(f"✅ Валидация пройдена успешно: {validation_results['overall_score']:.2f}%")

        return validation_results

# =================================================================
# NOTIFICATION UTILITIES (с улучшениями для Dynamic vLLM)
# =================================================================

class NotificationUtils:
    """Утилиты для отправки уведомлений"""

    @staticmethod
    def send_success_notification(context: Context, results: Dict[str, Any]):
        """Отправка уведомления об успешном завершении"""
        task_id = context['dag_run'].dag_id
        dag_run_id = context['dag_run'].run_id

        # Извлекаем информацию о моделях из результатов
        model_info = ""
        if 'transformation_stats' in results:
            model_info += f"\n- Content Model: {results['transformation_stats'].get('model_used', 'unknown')}"
        if 'translation_stats' in results:
            model_info += f"\n- Translation Model: {results['translation_stats'].get('model_used', 'unknown')}"

        message = f"""
✅ PDF конвейер v2.0 с динамической подгрузкой моделей завершен

Задача: {task_id}
Run ID: {dag_run_id}
Время выполнения: {context['dag_run'].end_date - context['dag_run'].start_date if context['dag_run'].end_date else 'N/A'}

🤖 Использованные модели:{model_info}

Результаты валидации:
- Общий балл качества: {results.get('overall_score', 'N/A')}%
- Валидация пройдена: {'✅ Да' if results.get('quality_passed') else '❌ Нет'}
"""

        logger.info(message)
        # Здесь можно добавить отправку в Slack, email и т.д.

    @staticmethod
    def send_failure_notification(context: Context, exception: Exception):
        """Отправка уведомления об ошибке"""
        task_id = context['task_instance'].task_id
        dag_id = context['dag_run'].dag_id

        message = f"""
❌ Ошибка в PDF конвейере v2.0 с динамической подгрузкой

DAG: {dag_id}
Task: {task_id}
Ошибка: {str(exception)}

Проверьте статус Dynamic vLLM Server: {os.getenv('VLLM_SERVER_URL', 'http://vllm-server:8000')}/v1/models/status
"""

        logger.error(message)

# =================================================================
# SHARED UTILITIES (обновлены)
# =================================================================

class SharedUtils:
    """Общие утилиты для всех DAG"""

    @staticmethod
    def prepare_output_path(filename: str, target_language: str, timestamp: int) -> str:
        """Подготовка пути для сохранения результата"""
        base_name = os.path.splitext(filename)[0]
        output_dir = f"/app/output/{target_language}"
        os.makedirs(output_dir, exist_ok=True)
        return f"{output_dir}/{timestamp}_{base_name}.md"

    @staticmethod
    def save_final_result(content: str, output_path: str, metadata: Dict = None):
        """Сохранение финального результата"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Сохранение метаданных
        if metadata:
            metadata_path = output_path.replace('.md', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        # Установка правильных прав доступа (1000:1000)
        try:
            os.chown(output_path, 1000, 1000)
            if metadata:
                os.chown(metadata_path, 1000, 1000)
        except:
            pass  # Игнорируем ошибки прав доступа в контейнере

    @staticmethod
    def validate_input_file(file_path: str) -> bool:
        """Валидация входного PDF файла"""
        if not os.path.exists(file_path):
            return False
            
        if not file_path.lower().endswith('.pdf'):
            return False
            
        # Проверка размера файла (макс 500MB)
        max_size = 500 * 1024 * 1024
        if os.path.getsize(file_path) > max_size:
            return False
            
        return True

    @staticmethod
    def get_processing_stats(start_time: datetime, end_time: datetime = None) -> Dict[str, Any]:
        """Получение статистики обработки"""
        end_time = end_time or datetime.now()
        duration = end_time - start_time
        
        return {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'duration_human': str(duration)
        }

    @staticmethod
    def check_vllm_server_health() -> Dict[str, Any]:
        """Проверка состояния Dynamic vLLM Server"""
        vllm_url = os.getenv('VLLM_SERVER_URL', 'http://vllm-server:8000')
        
        try:
            # Проверяем health endpoint
            response = requests.get(f"{vllm_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                
                # Получаем дополнительную информацию о моделях
                try:
                    models_response = requests.get(f"{vllm_url}/v1/models/status", timeout=10)
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        health_data.update(models_data)
                except:
                    pass
                
                return health_data
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

# =================================================================
# CONFIGURATION UTILITIES (обновлены для Dynamic vLLM)
# =================================================================

class ConfigUtils:
    """Утилиты для работы с конфигурацией"""

    @staticmethod
    def get_service_config() -> Dict[str, str]:
        """Получение конфигурации всех сервисов"""
        return {
            'vllm': os.getenv('VLLM_SERVER_URL', 'http://vllm-server:8000'),
            'document_processor': os.getenv('DOCUMENT_PROCESSOR_URL', 'http://document-processor:8001'),
            'qa_service': os.getenv('QA_SERVICE_ENDPOINT', 'http://quality-assurance:8002'),
            'translator': os.getenv('TRANSLATOR_URL', 'http://translator:8003')
        }

    @staticmethod
    def get_model_config() -> Dict[str, str]:
        """Получение конфигурации моделей для Dynamic vLLM"""
        return {
            'content_transformation_model': 'Qwen/Qwen2.5-VL-32B-Instruct',
            'translation_model': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
            'dynamic_loading_enabled': True,
            'vllm_endpoint': '/v1/chat/completions',
            'model_status_endpoint': '/v1/models/status',
            'model_swap_endpoint': '/v1/models/swap'
        }

    @staticmethod
    def get_processing_defaults() -> Dict[str, Any]:
        """Получение настроек по умолчанию для обработки"""
        return {
            'enable_ocr': True,
            'ocr_languages': 'chi_sim,eng,rus',
            'quality_target': 100.0,
            'auto_correct': True,
            'preserve_technical_terms': True,
            'max_file_size_mb': 500,
            'batch_size': 4,
            'disable_thinking': True,
            'dynamic_model_loading': True,
            'model_swap_timeout': 300,
            'vram_optimization': True
        }

    @staticmethod
    def get_dynamic_vllm_settings() -> Dict[str, Any]:
        """Настройки специфичные для Dynamic vLLM Server"""
        return {
            'auto_model_selection': True,
            'model_swap_timeout': 300,
            'memory_cleanup_aggressive': True,
            'health_check_interval': 30,
            'supported_tasks': ['content_transformation', 'translation'],
            'model_aliases': {
                'content-transformer': 'content_transformation',
                'translator': 'translation'
            }
        }