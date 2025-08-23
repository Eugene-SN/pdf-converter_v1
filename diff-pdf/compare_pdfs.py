# PDF Comparison Service
# Сервис для сравнения оригинального PDF с результатом конвейера

from flask import Flask, request, jsonify, send_file
import subprocess
import os
import tempfile
import shutil
import logging
from datetime import datetime
import json
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация сравнения
COMPARISON_CONFIG = {
    'supported_formats': ['pdf'],
    'output_formats': ['pdf', 'png', 'json'],
    'comparison_methods': ['visual', 'text', 'structure'],
    'tolerance_levels': ['strict', 'normal', 'relaxed']
}

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка состояния сервиса"""
    try:
        # Проверка доступности diff-pdf
        result = subprocess.run(['diff-pdf', '--version'], 
                              capture_output=True, text=True, timeout=5)
        
        diff_pdf_available = result.returncode == 0
        
        # Проверка poppler-utils
        poppler_result = subprocess.run(['pdftoppm', '-v'], 
                                      capture_output=True, text=True, timeout=5)
        poppler_available = poppler_result.returncode == 0
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'diff_pdf_available': diff_pdf_available,
            'poppler_available': poppler_available,
            'supported_methods': COMPARISON_CONFIG['comparison_methods'],
            'supported_formats': COMPARISON_CONFIG['supported_formats']
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/v1/compare', methods=['POST'])
def compare_pdfs():
    """Основной API для сравнения PDF файлов"""
    try:
        # Проверка наличия файлов
        if 'original' not in request.files or 'processed' not in request.files:
            return jsonify({'error': 'Требуются оба файла: original и processed'}), 400
        
        original_file = request.files['original']
        processed_file = request.files['processed']
        
        if original_file.filename == '' or processed_file.filename == '':
            return jsonify({'error': 'Файлы не выбраны'}), 400
        
        # Параметры сравнения
        comparison_method = request.form.get('method', 'visual')
        tolerance_level = request.form.get('tolerance', 'normal')
        output_format = request.form.get('output_format', 'json')
        
        if comparison_method not in COMPARISON_CONFIG['comparison_methods']:
            return jsonify({'error': f'Неподдерживаемый метод: {comparison_method}'}), 400
        
        # Создание временных файлов
        with tempfile.TemporaryDirectory() as temp_dir:
            # Сохранение файлов
            original_path = os.path.join(temp_dir, 'original.pdf')
            processed_path = os.path.join(temp_dir, 'processed.pdf')
            
            original_file.save(original_path)
            processed_file.save(processed_path)
            
            # Выполнение сравнения
            comparison_result = perform_comparison(
                original_path, 
                processed_path, 
                comparison_method, 
                tolerance_level,
                temp_dir
            )
            
            # Возврат результата в зависимости от формата
            if output_format == 'json':
                return jsonify(comparison_result)
            elif output_format == 'pdf' and comparison_result.get('diff_pdf_path'):
                return send_file(comparison_result['diff_pdf_path'], 
                               as_attachment=True,
                               download_name='comparison_result.pdf')
            elif output_format == 'png' and comparison_result.get('diff_images'):
                # Возврат первого изображения различий
                return send_file(comparison_result['diff_images'][0],
                               as_attachment=True,
                               download_name='comparison_page_1.png')
            else:
                return jsonify(comparison_result)
    
    except Exception as e:
        logger.error(f"Ошибка сравнения: {e}")
        return jsonify({'error': f'Внутренняя ошибка: {str(e)}'}), 500

def perform_comparison(original_path, processed_path, method, tolerance, temp_dir):
    """Выполнение сравнения PDF файлов"""
    
    comparison_result = {
        'timestamp': datetime.now().isoformat(),
        'method': method,
        'tolerance': tolerance,
        'files': {
            'original': original_path,
            'processed': processed_path
        },
        'differences_found': False,
        'similarity_score': 0.0,
        'page_count': {
            'original': 0,
            'processed': 0
        },
        'detailed_results': []
    }
    
    try:
        # Получение количества страниц
        original_pages = get_pdf_page_count(original_path)
        processed_pages = get_pdf_page_count(processed_path)
        
        comparison_result['page_count']['original'] = original_pages
        comparison_result['page_count']['processed'] = processed_pages
        
        if original_pages != processed_pages:
            comparison_result['differences_found'] = True
            comparison_result['page_count_mismatch'] = True
        
        # Выбор метода сравнения
        if method == 'visual':
            result = visual_comparison(original_path, processed_path, tolerance, temp_dir)
        elif method == 'text':
            result = text_comparison(original_path, processed_path, tolerance, temp_dir)
        elif method == 'structure':
            result = structure_comparison(original_path, processed_path, tolerance, temp_dir)
        else:
            result = visual_comparison(original_path, processed_path, tolerance, temp_dir)
        
        comparison_result.update(result)
        
        return comparison_result
        
    except Exception as e:
        comparison_result['error'] = str(e)
        return comparison_result

def get_pdf_page_count(pdf_path):
    """Получение количества страниц в PDF"""
    try:
        result = subprocess.run(['pdfinfo', pdf_path], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Pages:'):
                    return int(line.split(':')[1].strip())
        
        return 0
    except:
        return 0

def visual_comparison(original_path, processed_path, tolerance, temp_dir):
    """Визуальное сравнение PDF файлов"""
    try:
        # Использование diff-pdf для визуального сравнения
        diff_pdf_path = os.path.join(temp_dir, 'visual_diff.pdf')
        
        diff_command = [
            'diff-pdf',
            '--output-diff', diff_pdf_path,
            original_path,
            processed_path
        ]
        
        result = subprocess.run(diff_command, 
                              capture_output=True, 
                              text=True, 
                              timeout=60)
        
        differences_found = result.returncode != 0
        
        # Конвертация страниц в изображения для анализа
        similarity_scores = analyze_visual_similarity(original_path, processed_path, temp_dir)
        
        overall_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        visual_result = {
            'differences_found': differences_found,
            'similarity_score': float(overall_similarity),
            'page_similarities': [float(score) for score in similarity_scores],
            'diff_pdf_path': diff_pdf_path if os.path.exists(diff_pdf_path) else None,
            'visual_analysis': {
                'pages_analyzed': len(similarity_scores),
                'min_similarity': float(min(similarity_scores)) if similarity_scores else 0.0,
                'max_similarity': float(max(similarity_scores)) if similarity_scores else 0.0,
                'std_similarity': float(np.std(similarity_scores)) if similarity_scores else 0.0
            }
        }
        
        return visual_result
        
    except Exception as e:
        return {
            'differences_found': True,
            'similarity_score': 0.0,
            'error': f'Ошибка визуального сравнения: {str(e)}'
        }

def analyze_visual_similarity(original_path, processed_path, temp_dir):
    """Анализ визуального сходства страниц"""
    similarity_scores = []
    
    try:
        # Конвертация PDF в изображения
        original_images_dir = os.path.join(temp_dir, 'original_images')
        processed_images_dir = os.path.join(temp_dir, 'processed_images')
        
        os.makedirs(original_images_dir, exist_ok=True)
        os.makedirs(processed_images_dir, exist_ok=True)
        
        # Конвертация оригинала
        subprocess.run([
            'pdftoppm', '-png', '-r', '150',
            original_path, 
            os.path.join(original_images_dir, 'page')
        ], timeout=30)
        
        # Конвертация обработанного
        subprocess.run([
            'pdftoppm', '-png', '-r', '150', 
            processed_path,
            os.path.join(processed_images_dir, 'page')
        ], timeout=30)
        
        # Сравнение изображений
        original_images = sorted([f for f in os.listdir(original_images_dir) if f.endswith('.png')])
        processed_images = sorted([f for f in os.listdir(processed_images_dir) if f.endswith('.png')])
        
        min_pages = min(len(original_images), len(processed_images))
        
        for i in range(min_pages):
            original_img_path = os.path.join(original_images_dir, original_images[i])
            processed_img_path = os.path.join(processed_images_dir, processed_images[i])
            
            similarity = calculate_image_similarity(original_img_path, processed_img_path)
            similarity_scores.append(similarity)
        
        return similarity_scores
        
    except Exception as e:
        logger.error(f"Ошибка анализа визуального сходства: {e}")
        return []

def calculate_image_similarity(img1_path, img2_path):
    """Расчет сходства между двумя изображениями"""
    try:
        # Загрузка изображений
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Приведение к одному размеру
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        
        img1_resized = cv2.resize(img1, (width, height))
        img2_resized = cv2.resize(img2, (width, height))
        
        # Расчет SSIM (Structural Similarity Index)
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        # Простой расчет сходства через MSE
        mse = np.mean((gray1 - gray2) ** 2)
        
        if mse == 0:
            return 100.0
        else:
            # Нормализация к процентам
            max_mse = 255 ** 2
            similarity = (1 - (mse / max_mse)) * 100
            return max(0.0, similarity)
            
    except Exception as e:
        logger.error(f"Ошибка расчета сходства изображений: {e}")
        return 0.0

def text_comparison(original_path, processed_path, tolerance, temp_dir):
    """Текстовое сравнение PDF файлов"""
    try:
        # Извлечение текста из PDF
        original_text = extract_text_from_pdf(original_path)
        processed_text = extract_text_from_pdf(processed_path)
        
        # Сравнение текста
        text_similarity = calculate_text_similarity(original_text, processed_text)
        
        text_result = {
            'differences_found': text_similarity < 90.0,
            'similarity_score': text_similarity,
            'text_analysis': {
                'original_length': len(original_text),
                'processed_length': len(processed_text),
                'length_ratio': len(processed_text) / len(original_text) if len(original_text) > 0 else 0.0
            }
        }
        
        return text_result
        
    except Exception as e:
        return {
            'differences_found': True,
            'similarity_score': 0.0,
            'error': f'Ошибка текстового сравнения: {str(e)}'
        }

def extract_text_from_pdf(pdf_path):
    """Извлечение текста из PDF"""
    try:
        result = subprocess.run(['pdftotext', pdf_path, '-'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return result.stdout
        else:
            return ""
            
    except Exception:
        return ""

def calculate_text_similarity(text1, text2):
    """Расчет сходства текстов"""
    if not text1 and not text2:
        return 100.0
    
    if not text1 or not text2:
        return 0.0
    
    # Простой расчет на основе общих символов
    text1_clean = ''.join(text1.split()).lower()
    text2_clean = ''.join(text2.split()).lower()
    
    if len(text1_clean) == 0 and len(text2_clean) == 0:
        return 100.0
    
    # Расчет схожести через пересечение
    common_chars = sum(1 for c1, c2 in zip(text1_clean, text2_clean) if c1 == c2)
    max_length = max(len(text1_clean), len(text2_clean))
    
    similarity = (common_chars / max_length) * 100 if max_length > 0 else 0.0
    
    return similarity

def structure_comparison(original_path, processed_path, tolerance, temp_dir):
    """Структурное сравнение PDF файлов"""
    try:
        # Анализ структуры PDF (метаданные, закладки, аннотации)
        original_structure = analyze_pdf_structure(original_path)
        processed_structure = analyze_pdf_structure(processed_path)
        
        structure_similarity = compare_structures(original_structure, processed_structure)
        
        structure_result = {
            'differences_found': structure_similarity < 95.0,
            'similarity_score': structure_similarity,
            'structure_analysis': {
                'original': original_structure,
                'processed': processed_structure
            }
        }
        
        return structure_result
        
    except Exception as e:
        return {
            'differences_found': True,
            'similarity_score': 0.0,
            'error': f'Ошибка структурного сравнения: {str(e)}'
        }

def analyze_pdf_structure(pdf_path):
    """Анализ структуры PDF"""
    try:
        # Получение информации о PDF
        result = subprocess.run(['pdfinfo', pdf_path], 
                              capture_output=True, text=True, timeout=10)
        
        structure = {
            'pages': 0,
            'title': '',
            'author': '',
            'creator': '',
            'has_bookmarks': False,
            'has_annotations': False
        }
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Pages:'):
                    structure['pages'] = int(line.split(':')[1].strip())
                elif line.startswith('Title:'):
                    structure['title'] = line.split(':', 1)[1].strip()
                elif line.startswith('Author:'):
                    structure['author'] = line.split(':', 1)[1].strip()
                elif line.startswith('Creator:'):
                    structure['creator'] = line.split(':', 1)[1].strip()
        
        return structure
        
    except Exception:
        return {
            'pages': 0,
            'title': '',
            'author': '',
            'creator': '',
            'has_bookmarks': False,
            'has_annotations': False
        }

def compare_structures(struct1, struct2):
    """Сравнение структур PDF"""
    score = 100.0
    
    # Сравнение количества страниц
    if struct1['pages'] != struct2['pages']:
        score -= 20.0
    
    # Сравнение метаданных (менее критично)
    if struct1['title'] != struct2['title']:
        score -= 5.0
    
    if struct1['author'] != struct2['author']:
        score -= 5.0
    
    return max(0.0, score)

@app.route('/api/v1/batch_compare', methods=['POST'])
def batch_compare():
    """Пакетное сравнение множественных PDF файлов"""
    try:
        original_files = request.files.getlist('original_files')
        processed_files = request.files.getlist('processed_files')
        
        if len(original_files) != len(processed_files):
            return jsonify({'error': 'Количество оригинальных и обработанных файлов должно совпадать'}), 400
        
        comparison_method = request.form.get('method', 'visual')
        tolerance_level = request.form.get('tolerance', 'normal')
        
        results = []
        
        for orig_file, proc_file in zip(original_files, processed_files):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    orig_path = os.path.join(temp_dir, orig_file.filename)
                    proc_path = os.path.join(temp_dir, proc_file.filename)
                    
                    orig_file.save(orig_path)
                    proc_file.save(proc_path)
                    
                    comparison_result = perform_comparison(
                        orig_path, proc_path, comparison_method, tolerance_level, temp_dir
                    )
                    
                    results.append({
                        'original_file': orig_file.filename,
                        'processed_file': proc_file.filename,
                        'comparison': comparison_result
                    })
                    
            except Exception as e:
                results.append({
                    'original_file': orig_file.filename,
                    'processed_file': proc_file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'batch_id': int(datetime.now().timestamp()),
            'total_comparisons': len(results),
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'Ошибка пакетного сравнения: {str(e)}'}), 500

# Создание необходимых директорий при запуске
os.makedirs('/app/original', exist_ok=True)
os.makedirs('/app/compare', exist_ok=True)
os.makedirs('/app/temp', exist_ok=True)
os.makedirs('/app/logs', exist_ok=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)