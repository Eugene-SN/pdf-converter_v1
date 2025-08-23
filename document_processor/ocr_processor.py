#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Processor - PaddleOCR с поддержкой множественных языков.

- Автоматически скачивает недостающие модели в PADDLEX_HOME.
- Поддерживает CPU/GPU через конфиг.
- Корректно парсит результаты PaddleOCR.
- Опциональная кросс-валидация Tesseract для ch/en/ru.
- Совместим с текущим main.py (есть process_document_pages).
"""

from typing import List, Dict, Any, Union, Optional
import logging
from pathlib import Path

import pytesseract
from paddleocr import PaddleOCR

# Для PDF -> изображения
from pdf2image import convert_from_path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OCRConfig:
    def __init__(
        self,
        use_gpu: bool = False,
        lang: Union[str, List[str]] = "ch",
        confidence_threshold: float = 0.8,
        enable_tesseract_cv: bool = True,
        pdf_dpi: int = 200,
    ):
        self.use_gpu = use_gpu
        # Поддерживаем как строку, так и список языков, а также строку "ch,en,ru"
        if isinstance(lang, str):
            if "," in lang:
                self.lang = [l.strip() for l in lang.split(",") if l.strip()]
            else:
                self.lang = [lang.strip()]
        else:
            self.lang = lang or ["ch"]
        self.confidence_threshold = float(confidence_threshold)
        self.enable_tesseract_cv = bool(enable_tesseract_cv)
        self.pdf_dpi = int(pdf_dpi)

    def __repr__(self):
        return (
            f"OCRConfig(use_gpu={self.use_gpu}, "
            f"lang={self.lang}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"enable_tesseract_cv={self.enable_tesseract_cv}, "
            f"pdf_dpi={self.pdf_dpi})"
        )


class OCRProcessor:
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        logger.info("Initializing OCRProcessor with %s", self.config)

        # Выбор устройства
        device = "gpu:0" if self.config.use_gpu else "cpu"

        # Инициализируем отдельный движок для каждого языка
        self.ocr_engines: Dict[str, PaddleOCR] = {}
        self.supported_langs: List[str] = []

        for lang in self.config.lang:
            try:
                logger.info("Initializing PaddleOCR for language: %s", lang)
                # PaddleOCR сам скачает модели при первой инициализации, если их нет
                engine = PaddleOCR(
                    lang=lang,
                    use_textline_orientation=True, 
                )
                self.ocr_engines[lang] = engine
                self.supported_langs.append(lang)
                logger.info("PaddleOCR for %s initialized", lang)
            except Exception as e:
                logger.error("Failed to initialize PaddleOCR for %s: %s", lang, e)

        if not self.ocr_engines:
            raise RuntimeError("No PaddleOCR engines initialized for requested languages")

        self.confidence_threshold = self.config.confidence_threshold

        # Карта языков для Tesseract (минимально необходимая)
        self.tesseract_lang_map = {
            "ch": "chi_sim",
            "en": "eng",
            "ru": "rus",
        }

    def _resolve_lang(self, lang: Optional[str]) -> str:
        """Возвращает корректный язык, инициализированный в движке."""
        if not lang:
            return self.supported_langs[0]
        if lang not in self.ocr_engines:
            fallback = self.supported_langs
            logger.warning("Requested lang '%s' not initialized, fallback to '%s'", lang, fallback)
            return fallback
        return lang

    def ocr_image(self, image_path: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        OCR по пути до изображения. Возвращает список блоков:
        {
          "text": str,
          "confidence": float,
          "bbox": [[x,y], ... 4 точки],
          "engine": "paddle" | "tesseract",
          "language": lang
        }
        """
        try:
            eff_lang = self._resolve_lang(lang)
            engine = self.ocr_engines[eff_lang]

            result = engine.ocr(image_path)
            output: List[Dict[str, Any]] = []

            # result — список строк [(bbox, (text, conf)), ...]
            if result and len(result) > 0 and result:
                for line in result:
                    try:
                        bbox, (text, conf) = line
                        if conf >= self.confidence_threshold:
                            output.append(
                                {
                                    "text": text,
                                    "confidence": float(conf),
                                    "bbox": bbox,
                                    "engine": "paddle",
                                    "language": eff_lang,
                                }
                            )
                    except Exception as ie:
                        logger.debug("Skip malformed OCR line: %s (%s)", line, ie)

            # Кросс-валидация Tesseract (опционально)
            if self.config.enable_tesseract_cv and eff_lang in self.tesseract_lang_map:
                try:
                    t_lang = self.tesseract_lang_map[eff_lang]
                    t_text = pytesseract.image_to_string(image_path, lang=t_lang).strip()
                    if t_text:
                        output.append(
                            {
                                "text": t_text,
                                "confidence": 0.9,  # эвристика
                                "bbox": None,
                                "engine": "tesseract",
                                "language": eff_lang,
                            }
                        )
                except Exception as te:
                    logger.debug("Tesseract failed for %s: %s", eff_lang, te)

            return output

        except Exception as e:
            logger.error("Error during OCR on %s (lang=%s): %s", image_path, lang, e)
            return []

    async def process_document_pages(self, pdf_path: str, work_dir: str, lang: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Совместимо с main.py:
        - Конвертирует PDF в изображения (pdf2image, DPI из конфигурации)
        - Прогоняет OCR по каждой странице (по выбранному языку или первому доступному)
        - Возвращает список результатов по страницам
        """
        results: List[Dict[str, Any]] = []
        eff_lang = self._resolve_lang(lang)

        try:
            Path(work_dir).mkdir(parents=True, exist_ok=True)
            images: List[Image.Image] = convert_from_path(pdf_path, self.config.pdf_dpi)
            for idx, pil_img in enumerate(images, start=1):
                img_path = Path(work_dir) / f"page_{idx}.png"
                pil_img.save(str(img_path))

                page_ocr = self.ocr_image(str(img_path), eff_lang)
                results.append(
                    {
                        "page": idx,
                        "language": eff_lang,
                        "ocr_results": page_ocr,
                        "image_path": str(img_path),
                        "text_count": len(page_ocr),
                    }
                )
        except Exception as e:
            logger.error("Error in process_document_pages: %s", e)

        return results

    def ocr_image_multilang(self, image_path: str, langs: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """OCR по нескольким языкам. Возвращает словарь lang -> список блоков."""
        langs = langs or self.supported_langs
        results: Dict[str, List[Dict[str, Any]]] = {}
        for l in langs:
            if l in self.supported_langs:
                results[l] = self.ocr_image(image_path, l)
            else:
                logger.debug("Language %s not initialized, skip", l)
        return results
