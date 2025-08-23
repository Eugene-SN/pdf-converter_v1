#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Docling Processor для PDF Converter Pipeline v4.0
Исправленная версия с правильным использованием Docling API
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import tempfile
import json
from dataclasses import dataclass
from datetime import datetime

# Docling импорты
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# Вспомогательные библиотеки
import structlog
import numpy as np
from PIL import Image
import pandas as pd
import re

# Для работы с файловой системой
import shutil
from pathlib import Path

# Prometheus метрики
from prometheus_client import Counter, Histogram, Gauge

# =======================================================================================
# КОНФИГУРАЦИЯ И МЕТРИКИ
# =======================================================================================

logger = structlog.get_logger("docling_processor")

# Метрики Prometheus
docling_requests = Counter('docling_requests_total', 'Total Docling processing requests', ['status'])
docling_duration = Histogram('docling_processing_duration_seconds', 'Docling processing duration')
docling_pages = Histogram('docling_pages_processed', 'Pages processed by Docling')
docling_elements = Counter('docling_elements_extracted', 'Elements extracted by type', ['element_type'])

@dataclass
class DoclingConfig:
    """Конфигурация Docling процессора"""
    model_path: str = "/mnt/storage/models/shared/docling"
    use_gpu: bool = False
    max_workers: int = 4
    timeout: int = 300
    cache_dir: str = "/app/cache"
    temp_dir: str = "/app/temp"
    
    # Параметры обработки
    extract_images: bool = True
    extract_tables: bool = True
    extract_formulas: bool = True
    high_quality_ocr: bool = True
    
    # Специфические настройки для китайских документов
    chinese_language_support: bool = True
    preserve_chinese_layout: bool = True
    mixed_language_mode: bool = True

# =======================================================================================
# ОСНОВНЫЕ ТИПЫ ДАННЫХ
# =======================================================================================

@dataclass
class ProcessedElement:
    """Обработанный элемент документа"""
    element_type: str
    content: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    page_number: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DocumentStructure:
    """Структура документа после обработки"""
    title: Optional[str] = None
    authors: List[str] = None
    sections: List[Dict[str, Any]] = None
    tables: List[Dict[str, Any]] = None
    images: List[Dict[str, Any]] = None
    formulas: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.sections is None:
            self.sections = []
        if self.tables is None:
            self.tables = []
        if self.images is None:
            self.images = []
        if self.formulas is None:
            self.formulas = []
        if self.metadata is None:
            self.metadata = {}

# =======================================================================================
# DOCLING PROCESSOR КЛАСС  
# =======================================================================================

class DoclingProcessor:
    """Главный класс для обработки PDF через Docling"""

    def __init__(self, config: Optional[DoclingConfig] = None):
        self.config = config or DoclingConfig()
        self.use_ocr = getattr(self.config, 'use_ocr', True)
        self.converter: Optional[DocumentConverter] = None
        self.logger = structlog.get_logger("docling_processor")

        # Создаем необходимые директории
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)

        self._initialize_converter()

    def _initialize_converter(self):
        """Инициализация Docling конвертера"""
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import PdfFormatOption

            # ✅ УПРОЩЕННАЯ ИНИЦИАЛИЗАЦИЯ
            pdf_format_options = PdfFormatOption(
                backend=PyPdfiumDocumentBackend,
                pipeline_options=PdfPipelineOptions(
                    enable_layout_analysis=True,
                    enable_ocr=self.use_ocr,
                    images_scale=1.0,
                    generate_page_images=False,
                    generate_table_images=False,
                    generate_picture_images=False
                )
            )

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: pdf_format_options
                }
            )

            self.logger.info("Docling converter initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Docling converter: {e}")
            raise

    async def process_document(self, pdf_path: str, output_dir: str, use_ocr: bool = True) -> DocumentStructure:
        """
        Основной метод обработки PDF документа
        """
        start_time = datetime.now()
        
        try:
            docling_requests.labels(status='started').inc()

            # Проверяем существование файла
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Создаем временную директорию для обработки
            with tempfile.TemporaryDirectory(dir=self.config.temp_dir) as temp_dir:
                # Конвертируем документ
                self.logger.info(f"Starting Docling conversion for: {pdf_path}")
                conv_result = self.converter.convert(pdf_path)
                # Добавьте проверку OCR
                self.use_ocr = use_ocr
                logger.info(f"📥 OCR setting: {use_ocr}")
                logger.info(f"▶ Docling initialised with OCR: {self.use_ocr}")

                # ✅ НОВАЯ ДИАГНОСТИКА - посмотрим ВСЕ атрибуты PageItem
                self.logger.info(f"🔍 DEBUG: Document pages type: {type(conv_result.document.pages)}")
                self.logger.info(f"🔍 DEBUG: Document pages count: {len(conv_result.document.pages)}")
                
                # Проверяем ВСЕ атрибуты первой страницы
                if conv_result.document.pages:
                    first_page_key = list(conv_result.document.pages.keys())[0]
                    first_page = conv_result.document.pages[first_page_key]
                    all_attrs = [attr for attr in dir(first_page) if not attr.startswith('_')]
                    self.logger.info(f"🔍 DEBUG: Page {first_page_key} ALL attributes: {all_attrs}")
                    
                    # Проверим специфические атрибуты
                    for attr in ['elements', 'items', 'content', 'text', 'layout', 'blocks']:
                        if hasattr(first_page, attr):
                            self.logger.info(f"🔍 DEBUG: Page has attribute '{attr}': {getattr(first_page, attr, None)}")

                # ✅ НОВЫЙ ПОДХОД - используем export методы документа
                document_structure = await self._extract_document_structure_new(
                    conv_result, temp_dir, output_dir
                )

                # Обновляем метрики
                duration = (datetime.now() - start_time).total_seconds()
                docling_duration.observe(duration)
                docling_pages.observe(len(conv_result.document.pages))
                docling_requests.labels(status='completed').inc()

                self.logger.info(
                    f"Document processed successfully in {duration:.2f}s",
                    pages=len(conv_result.document.pages),
                    elements=len(document_structure.sections) + len(document_structure.tables)
                )

                return document_structure

        except Exception as e:
            docling_requests.labels(status='error').inc()
            self.logger.error(f"Error processing document {pdf_path}: {e}")
            raise

    async def _extract_document_structure_new(
        self,
        conv_result,
        temp_dir: str,
        output_dir: str
    ) -> DocumentStructure:
        """✅ НОВАЯ ВЕРСИЯ извлечения структуры с правильным Docling API"""
        
        document = conv_result.document
        structure = DocumentStructure()

        # Извлечение метаданных
        structure.metadata = {
            "total_pages": len(document.pages),
            "processing_time": datetime.now().isoformat(),
            "docling_version": "1.5.0",
            "language_detected": "mixed"
        }

        try:
            # ✅ МЕТОД 1: Попробуем экспорт в текст
            try:
                full_text = document.export_to_text()
                if full_text and len(full_text.strip()) > 10:
                    self.logger.info(f"✅ SUCCESS: Extracted {len(full_text)} chars via export_to_text")
                    structure.sections = await self._parse_exported_text(full_text)
                    structure.title = await self._extract_title_from_text(full_text)
                else:
                    self.logger.warning("export_to_text returned empty or short text")
            except Exception as e:
                self.logger.warning(f"export_to_text failed: {e}")

            # ✅ МЕТОД 2: Попробуем экспорт в Markdown
            if not structure.sections:
                try:
                    markdown_text = document.export_to_markdown()
                    if markdown_text and len(markdown_text.strip()) > 10:
                        self.logger.info(f"✅ SUCCESS: Extracted {len(markdown_text)} chars via export_to_markdown")
                        structure.sections = await self._parse_markdown_content(markdown_text)
                        if not structure.title:
                            structure.title = await self._extract_title_from_text(markdown_text)
                    else:
                        self.logger.warning("export_to_markdown returned empty or short text")
                except Exception as e:
                    self.logger.warning(f"export_to_markdown failed: {e}")

            # ✅ МЕТОД 3: Попробуем прямой доступ к content страниц
            if not structure.sections:
                structure.sections = await self._extract_via_page_content(document)

            # ✅ МЕТОД 4: Fallback - создаем базовую секцию
            if not structure.sections:
                structure.sections = [{
                    'title': 'Document Content',
                    'level': 1,
                    'page': 1,
                    'content': f'Processed document with {len(document.pages)} pages using Docling',
                    'subsections': []
                }]

            # Пытаемся извлечь таблицы и изображения если возможно
            if self.config.extract_tables:
                structure.tables = await self._extract_tables_new(document, temp_dir, output_dir)

            if self.config.extract_images:
                structure.images = await self._extract_images_new(document, temp_dir, output_dir)

            self.logger.info(f"Final extraction result: {len(structure.sections)} sections, {len(structure.tables)} tables, {len(structure.images)} images")

        except Exception as e:
            self.logger.error(f"Error in new document structure extraction: {e}")
            # Создаем минимальную структуру
            structure.sections = [{
                'title': 'Error Processing Document',
                'level': 1,
                'page': 1,
                'content': f'Document processing encountered errors: {str(e)}',
                'subsections': []
            }]

        return structure

    async def _parse_exported_text(self, full_text: str) -> List[Dict[str, Any]]:
        """Парсинг экспортированного текста в секции"""
        sections = []
        
        try:
            # Простое разбиение на абзацы
            paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
            
            current_section = None
            
            for i, paragraph in enumerate(paragraphs):
                # Проверяем, является ли параграф заголовком
                if self._looks_like_heading(paragraph):
                    # Сохраняем предыдущую секцию
                    if current_section:
                        sections.append(current_section)
                    
                    # Создаем новую секцию
                    current_section = {
                        'title': paragraph[:200],  # Ограничиваем длину заголовка
                        'level': self._get_heading_level_from_text(paragraph),
                        'page': 1,  # Приблизительно
                        'content': '',
                        'subsections': []
                    }
                else:
                    # Добавляем контент к текущей секции
                    if current_section:
                        current_section['content'] += paragraph + '\n\n'
                    else:
                        # Создаем секцию без заголовка
                        current_section = {
                            'title': 'Document Content',
                            'level': 1,
                            'page': 1,
                            'content': paragraph + '\n\n',
                            'subsections': []
                        }
            
            # Добавляем последнюю секцию
            if current_section:
                sections.append(current_section)
            
            # Если не нашли ни одной секции, создаем одну с всем текстом
            if not sections:
                sections = [{
                    'title': 'Document Content',
                    'level': 1,
                    'page': 1,
                    'content': full_text,
                    'subsections': []
                }]
            
            self.logger.info(f"Parsed {len(sections)} sections from exported text")
            
        except Exception as e:
            self.logger.error(f"Error parsing exported text: {e}")
            sections = [{
                'title': 'Document Content',
                'level': 1,
                'page': 1,
                'content': full_text,
                'subsections': []
            }]
        
        return sections

    async def _parse_markdown_content(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Парсинг Markdown контента в секции"""
        sections = []
        
        try:
            lines = markdown_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Проверяем заголовки Markdown
                if line.startswith('#'):
                    # Сохраняем предыдущую секцию
                    if current_section:
                        sections.append(current_section)
                    
                    # Определяем уровень заголовка
                    level = 1
                    while line.startswith('#' * (level + 1)):
                        level += 1
                    
                    title = line.lstrip('#').strip()
                    
                    current_section = {
                        'title': title,
                        'level': level,
                        'page': 1,
                        'content': '',
                        'subsections': []
                    }
                else:
                    # Добавляем контент к текущей секции
                    if current_section:
                        current_section['content'] += line + '\n'
                    else:
                        # Создаем секцию без заголовка
                        current_section = {
                            'title': 'Document Content',
                            'level': 1,
                            'page': 1,
                            'content': line + '\n',
                            'subsections': []
                        }
            
            # Добавляем последнюю секцию
            if current_section:
                sections.append(current_section)
            
            if not sections:
                sections = [{
                    'title': 'Document Content',
                    'level': 1,
                    'page': 1,
                    'content': markdown_text,
                    'subsections': []
                }]
            
            self.logger.info(f"Parsed {len(sections)} sections from markdown")
            
        except Exception as e:
            self.logger.error(f"Error parsing markdown: {e}")
            sections = [{
                'title': 'Document Content',
                'level': 1,
                'page': 1,
                'content': markdown_text,
                'subsections': []
            }]
        
        return sections

    async def _extract_via_page_content(self, document) -> List[Dict[str, Any]]:
        """Извлечение через прямой доступ к контенту страниц"""
        sections = []
        
        try:
            for page_num, page in document.pages.items():
                # Проверяем различные возможные атрибуты
                page_content = ""
                
                for attr in ['content', 'text', 'body', 'data']:
                    if hasattr(page, attr):
                        content = getattr(page, attr)
                        if content and str(content).strip():
                            page_content = str(content)
                            break
                
                if page_content:
                    sections.append({
                        'title': f'Page {page_num}',
                        'level': 1,
                        'page': page_num,
                        'content': page_content,
                        'subsections': []
                    })
            
            self.logger.info(f"Extracted {len(sections)} sections via page content")
            
        except Exception as e:
            self.logger.error(f"Error extracting via page content: {e}")
        
        return sections

    async def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Извлечение заголовка из текста"""
        try:
            lines = text.split('\n')
            for line in lines[:10]:  # Проверяем первые 10 строк
                line = line.strip()
                if line and len(line) > 5 and len(line) < 200:
                    # Убираем Markdown заголовки
                    title = line.lstrip('#').strip()
                    if title:
                        return title
            return None
        except Exception:
            return None

    def _looks_like_heading(self, text: str) -> bool:
        """Проверка, выглядит ли текст как заголовок"""
        text = text.strip()
        
        # Слишком длинный или короткий текст
        if len(text) < 3 or len(text) > 200:
            return False
        
        # Заканчивается двоеточием
        if text.endswith(':'):
            return True
        
        # Содержит цифры в начале (нумерация)
        if re.match(r'^\d+[\.\)]\s+', text):
            return True
        
        # Весь в верхнем регистре и короткий
        if text.isupper() and len(text) < 100:
            return True
        
        # Содержит ключевые слова заголовков
        heading_keywords = ['章', '节', '部分', '第', '篇', 'chapter', 'section', 'part']
        if any(keyword in text.lower() for keyword in heading_keywords):
            return True
        
        return False

    def _get_heading_level_from_text(self, text: str) -> int:
        """Определение уровня заголовка из текста"""
        # Если начинается с цифры - уровень 1
        if re.match(r'^\d+[\.\)]', text):
            return 1
        
        # Если содержит подуровни (1.1, 1.1.1)
        if re.match(r'^\d+\.\d+', text):
            return 2
        
        if re.match(r'^\d+\.\d+\.\d+', text):
            return 3
        
        # По умолчанию
        return 1

    async def _extract_tables_new(self, document, temp_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """Новое извлечение таблиц"""
        # Пока возвращаем пустой список - таблицы можно будет добавить позже
        return []

    async def _extract_images_new(self, document, temp_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """Новое извлечение изображений"""
        # Пока возвращаем пустой список - изображения можно будет добавить позже  
        return []

    # Остальные методы остаются без изменений
    async def _extract_title(self, document) -> Optional[str]:
        """Извлечение заголовка документа - старый метод для совместимости"""
        return None

    async def _extract_sections(self, document) -> List[Dict[str, Any]]:
        """Старый метод извлечения секций - для совместимости"""
        return []

    def _get_heading_level(self, element_type: str) -> int:
        """Определение уровня заголовка"""
        level_map = {
            'title': 1,
            'h1': 1,
            'heading': 1,
            'h2': 2,
            'h3': 3,
        }
        return level_map.get(element_type, 1)

    async def _extract_tables(self, document, temp_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """Старое извлечение таблиц"""
        return []

    async def _extract_images(self, document, temp_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """Старое извлечение изображений"""
        return []

    async def _extract_formulas(self, document) -> List[Dict[str, Any]]:
        """Извлечение математических формул"""
        return []

    def export_to_markdown(self, structure: DocumentStructure, output_path: str) -> str:
        """Экспорт структуры документа в Markdown"""
        try:
            md_content = []

            # Заголовок документа
            if structure.title:
                md_content.append(f"# {structure.title}\n")

            # Разделы
            for section in structure.sections:
                # Заголовок раздела
                level_prefix = "#" * section['level']
                md_content.append(f"{level_prefix} {section['title']}\n")

                # Содержание раздела
                if section['content']:
                    md_content.append(f"{section['content']}\n")

            # Таблицы
            if structure.tables:
                md_content.append("## Tables\n")
                for table in structure.tables:
                    md_content.append(f"### Table {table['id']} (Page {table['page']})\n")
                    md_content.append(f"Rows: {table['rows']}, Columns: {table['columns']}\n")
                    md_content.append(f"File: `{table['file_path']}`\n\n")

            # Изображения
            if structure.images:
                md_content.append("## Images\n")
                for image in structure.images:
                    md_content.append(f"### Image {image['id']} (Page {image['page']})\n")
                    md_content.append(f"![Image {image['id']}]({image['file_path']})\n")
                    if image.get('caption'):
                        md_content.append(f"*{image['caption']}*\n\n")
                    else:
                        md_content.append("\n")

            # Формулы
            if structure.formulas:
                md_content.append("## Formulas\n")
                for formula in structure.formulas:
                    md_content.append(f"### Formula {formula['id']} (Page {formula['page']})\n")
                    md_content.append(f"``````\n\n")

            # Сохранение в файл
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(md_content))

            self.logger.info(f"Markdown exported to: {output_path}")

            return '\n'.join(md_content)

        except Exception as e:
            self.logger.error(f"Error exporting to markdown: {e}")
            raise

# =======================================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =======================================================================================

def create_docling_processor(config: Optional[DoclingConfig] = None) -> DoclingProcessor:
    """Фабричная функция для создания Docling процессора"""
    return DoclingProcessor(config)

async def process_pdf_with_docling(
    pdf_path: str,
    output_dir: str,
    config: Optional[DoclingConfig] = None
) -> DocumentStructure:
    """
    Высокоуровневая функция для обработки PDF
    """
    processor = create_docling_processor(config)
    return await processor.process_document(pdf_path, output_dir)

# =======================================================================================
# ОСНОВНОЙ БЛОК ДЛЯ ТЕСТИРОВАНИЯ
# =======================================================================================

if __name__ == "__main__":
    async def main():
        config = DoclingConfig()
        processor = DoclingProcessor(config)

        pdf_path = "/app/temp/test_document.pdf"
        output_dir = "/app/temp/output"

        if Path(pdf_path).exists():
            structure = await processor.process_document(pdf_path, output_dir)

            md_path = Path(output_dir) / "document.md"
            processor.export_to_markdown(structure, str(md_path))

            print(f"Document processed successfully!")
            print(f"Sections: {len(structure.sections)}")
            print(f"Tables: {len(structure.tables)}")
            print(f"Images: {len(structure.images)}")
        else:
            print(f"Test file not found: {pdf_path}")

    asyncio.run(main())
