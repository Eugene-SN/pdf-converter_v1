# Flask API - Исправленный главный интерфейс управления PDF конвейером v2.0
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import requests
import os
import json
import time
from datetime import datetime
import logging
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', '/app/temp')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-key-change-in-production')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Исправленная конфигурация сервисов для v2.0
SERVICES = {
    'airflow': os.getenv('AIRFLOW_BASE_URL', 'http://airflow-webserver:8080'),
    'document_processor': os.getenv('DOCUMENT_PROCESSOR_URL', 'http://document-processor:8001'),
    'vllm': os.getenv('VLLM_BASE_URL', 'http://vllm-server:8000'),
    'quality_assurance': os.getenv('QUALITY_ASSURANCE_URL', 'http://quality-assurance:8002'),
    'translator': os.getenv('TRANSLATOR_URL', 'http://translator:8003')
}

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Простая проверка состояния Flask API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.0',
        'service': 'flask-api'
    }), 200

@app.route('/health/detailed', methods=['GET'])  
def detailed_health_check():
    """Детальная проверка всех сервисов"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.0',
        'flask_app': 'running',
        'services': {}
    }
    
    for service_name, url in SERVICES.items():
        try:
            response = requests.get(f"{url}/health", timeout=2)
            status['services'][service_name] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds(),
                'url': url
            }
        except Exception as e:
            status['services'][service_name] = {
                'status': 'error',
                'error': str(e),
                'url': url
            }
    
    # Определяем общий статус системы
    unhealthy_services = [name for name, info in status['services'].items()
                         if info['status'] != 'healthy']
    
    if unhealthy_services:
        status['status'] = 'degraded'
        status['unhealthy_services'] = unhealthy_services
        status['note'] = 'Flask API is running, but some external services are unavailable'
    
    # Возвращаем статус с соответствующим HTTP кодом для детального check'а
    return jsonify(status), 200 if status['status'] in ['healthy', 'degraded'] else 503

@app.route('/api/v1/convert', methods=['POST'])
def convert_pdf():
    """
    Основной API для конвертации PDF в Markdown
    Поддерживает модульную архитектуру DAG v2.0
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден в запросе'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Недопустимый файл. Поддерживаются только PDF файлы'}), 400
    
    # Параметры обработки
    target_language = request.form.get('target_language', 'ru')  # ru, en, zh, original
    quality_level = request.form.get('quality_level', 'high')   # basic, high, maximum
    enable_ocr = request.form.get('enable_ocr', 'true').lower() == 'true'
    preserve_structure = request.form.get('preserve_structure', 'true').lower() == 'true'
    enable_qa = request.form.get('enable_qa', 'true').lower() == 'true'
    
    # Сохранение файла
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    unique_filename = f"{timestamp}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    # Создание директории если не существует
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    file.save(file_path)
    
    # Конфигурация задачи для модульных DAG
    task_config = {
        'input_file': file_path,
        'filename': filename,
        'target_language': target_language,
        'quality_level': quality_level,
        'enable_ocr': enable_ocr,
        'preserve_structure': preserve_structure,
        'enable_qa': enable_qa,
        'timestamp': timestamp,
        'processing_chain': get_processing_chain(target_language, enable_qa)
    }
    
    # Запуск orchestrator DAG через Airflow API
    try:
        airflow_url = f"{SERVICES['airflow']}/api/v1/dags/orchestrator_dag/dagRuns"
        dag_run_id = f"pdf_convert_{timestamp}"
        
        logger.info(f"Запуск обработки файла {filename} с ID: {dag_run_id}")
        
        response = requests.post(
            airflow_url,
            json={
                'conf': task_config,
                'dag_run_id': dag_run_id
            },
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Basic YWRtaW46YWRtaW4='  # admin:admin в base64
            },
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            dag_run_info = response.json()
            return jsonify({
                'task_id': timestamp,
                'status': 'started',
                'dag_run_id': dag_run_id,
                'filename': filename,
                'target_language': target_language,
                'estimated_time': get_estimated_time(quality_level, target_language),
                'tracking_url': f"/api/v1/status/{timestamp}",
                'config': task_config
            }), 202
        else:
            logger.error(f"Ошибка запуска DAG: {response.status_code} - {response.text}")
            return jsonify({
                'error': 'Ошибка запуска задачи обработки',
                'details': response.text
            }), 500
    
    except Exception as e:
        logger.error(f"Ошибка связи с Airflow: {e}")
        return jsonify({
            'error': 'Сервис обработки временно недоступен',
            'details': str(e)
        }), 503

def get_processing_chain(target_language, enable_qa):
    """Определяет цепочку обработки DAG"""
    chain = ['document_preprocessing', 'content_transformation']
    
    if target_language != 'original':
        chain.append('translation_pipeline')
    
    if enable_qa:
        chain.append('quality_assurance')
    
    return chain

def get_estimated_time(quality_level, target_language):
    """Оценка времени обработки"""
    base_time = {
        'basic': '2-5 минут',
        'high': '5-15 минут', 
        'maximum': '15-30 минут'
    }.get(quality_level, '5-15 минут')
    
    if target_language != 'original':
        return f"{base_time} (включая перевод)"
    
    return base_time

@app.route('/api/v1/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Получение статуса задачи"""
    try:
        dag_run_id = f"pdf_convert_{task_id}"
        
        # Проверяем статус DAG run
        response = requests.get(
            f"{SERVICES['airflow']}/api/v1/dags/orchestrator_dag/dagRuns/{dag_run_id}",
            headers={'Authorization': 'Basic YWRtaW46YWRtaW4='},
            timeout=10
        )
        
        if response.status_code == 200:
            dag_info = response.json()
            state = dag_info.get('state', 'unknown')
            
            # Получаем детальную информацию о task'ах
            tasks_response = requests.get(
                f"{SERVICES['airflow']}/api/v1/dags/orchestrator_dag/dagRuns/{dag_run_id}/taskInstances",
                headers={'Authorization': 'Basic YWRtaW46YWRtaW4='},
                timeout=10
            )
            
            task_details = []
            current_step = None
            total_steps = 0
            completed_steps = 0
            
            if tasks_response.status_code == 200:
                tasks = tasks_response.json().get('task_instances', [])
                total_steps = len(tasks)
                
                for task in tasks:
                    task_state = task.get('state')
                    task_info = {
                        'task_id': task.get('task_id'),
                        'state': task_state,
                        'start_date': task.get('start_date'),
                        'end_date': task.get('end_date'),
                        'duration': task.get('duration')
                    }
                    task_details.append(task_info)
                    
                    if task_state == 'success':
                        completed_steps += 1
                    elif task_state == 'running':
                        current_step = task.get('task_id')
            
            # Прогресс обработки
            progress = (completed_steps / total_steps * 100) if total_steps > 0 else 0
            
            result = {
                'task_id': task_id,
                'status': state,
                'progress': round(progress, 1),
                'current_step': current_step,
                'completed_steps': completed_steps,
                'total_steps': total_steps,
                'start_time': dag_info.get('start_date'),
                'end_time': dag_info.get('end_date'),
                'task_details': task_details
            }
            
            # Если задача завершена успешно, добавляем ссылки на результаты
            if state == 'success':
                result['download_url'] = f"/api/v1/download/{task_id}"
                result['quality_report'] = f"/api/v1/quality-report/{task_id}"
            elif state == 'failed':
                result['error_details'] = "Ошибка обработки. Проверьте логи в Airflow UI"
            
            return jsonify(result)
        elif response.status_code == 404:
            return jsonify({'error': 'Задача не найдена'}), 404
        else:
            return jsonify({'error': f'Ошибка получения статуса: {response.status_code}'}), 500
    
    except Exception as e:
        logger.error(f"Ошибка получения статуса: {e}")
        return jsonify({'error': 'Ошибка получения статуса задачи'}), 500

@app.route('/api/v1/download/<task_id>', methods=['GET'])
def download_result(task_id):
    """Скачивание результата обработки"""
    output_dir = os.getenv('OUTPUT_DIR', '/app/output')
    
    # Поиск файлов результата
    possible_files = [
        f"{output_dir}/{task_id}.md",
        f"{output_dir}/{task_id}.zip",
        f"{output_dir}/pdf_convert_{task_id}.md",
        f"{output_dir}/pdf_convert_{task_id}.zip"
    ]
    
    for result_path in possible_files:
        if os.path.exists(result_path):
            return send_file(result_path, as_attachment=True)
    
    return jsonify({'error': 'Файл результата не найден'}), 404

@app.route('/api/v1/quality-report/<task_id>', methods=['GET'])
def get_quality_report(task_id):
    """Получение отчета о качестве обработки"""
    try:
        response = requests.get(
            f"{SERVICES['quality_assurance']}/api/report/{task_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return jsonify(response.json())
        elif response.status_code == 404:
            return jsonify({'error': 'Отчет о качестве не найден'}), 404
        else:
            return jsonify({'error': f'Ошибка получения отчета: {response.status_code}'}), 500
    
    except Exception as e:
        logger.error(f"Ошибка получения отчета качества: {e}")
        return jsonify({'error': 'Ошибка получения отчета о качестве'}), 500

@app.route('/api/v1/batch/convert', methods=['POST'])
def batch_convert():
    """Пакетная обработка нескольких PDF файлов"""
    files = request.files.getlist('files')
    if not files or len(files) == 0:
        return jsonify({'error': 'Файлы не найдены в запросе'}), 400
    
    target_language = request.form.get('target_language', 'ru')
    quality_level = request.form.get('quality_level', 'high')
    enable_qa = request.form.get('enable_qa', 'true').lower() == 'true'
    
    batch_id = int(time.time())
    task_ids = []
    failed_files = []
    
    for i, file in enumerate(files):
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                unique_filename = f"{batch_id}_{i}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                task_config = {
                    'input_file': file_path,
                    'filename': filename,
                    'target_language': target_language,
                    'quality_level': quality_level,
                    'enable_qa': enable_qa,
                    'batch_id': batch_id,
                    'batch_mode': True,
                    'batch_index': i
                }
                
                task_id = f"batch_{batch_id}_{i}"
                task_ids.append({
                    'task_id': task_id,
                    'filename': filename,
                    'status': 'queued'
                })
                
                # Запуск DAG для каждого файла
                requests.post(
                    f"{SERVICES['airflow']}/api/v1/dags/orchestrator_dag/dagRuns",
                    json={
                        'conf': task_config,
                        'dag_run_id': task_id
                    },
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': 'Basic YWRtaW46YWRtaW4='
                    },
                    timeout=30
                )
                
            except Exception as e:
                failed_files.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        else:
            failed_files.append({
                'filename': file.filename if file else 'unknown',
                'error': 'Недопустимый тип файла'
            })
    
    return jsonify({
        'batch_id': batch_id,
        'task_ids': task_ids,
        'failed_files': failed_files,
        'status': 'started',
        'total_files': len(task_ids),
        'failed_count': len(failed_files),
        'batch_status_url': f"/api/v1/batch/status/{batch_id}"
    }), 202

@app.route('/api/v1/batch/status/<batch_id>', methods=['GET'])
def get_batch_status(batch_id):
    """Получение статуса пакетной обработки"""
    try:
        # Поиск всех задач пакета
        response = requests.get(
            f"{SERVICES['airflow']}/api/v1/dags/orchestrator_dag/dagRuns",
            params={'limit': 100},
            headers={'Authorization': 'Basic YWRtaW46YWRtaW4='},
            timeout=10
        )
        
        if response.status_code == 200:
            dag_runs = response.json().get('dag_runs', [])
            batch_tasks = [run for run in dag_runs 
                          if run.get('dag_run_id', '').startswith(f'batch_{batch_id}_')]
            
            batch_status = {
                'batch_id': batch_id,
                'total_tasks': len(batch_tasks),
                'completed': 0,
                'failed': 0,
                'running': 0,
                'tasks': []
            }
            
            for task in batch_tasks:
                state = task.get('state', 'unknown')
                task_info = {
                    'task_id': task.get('dag_run_id'),
                    'state': state,
                    'start_date': task.get('start_date'),
                    'end_date': task.get('end_date')
                }
                batch_status['tasks'].append(task_info)
                
                if state == 'success':
                    batch_status['completed'] += 1
                elif state == 'failed':
                    batch_status['failed'] += 1
                elif state in ['running', 'queued']:
                    batch_status['running'] += 1
            
            # Общий статус пакета
            if batch_status['completed'] == batch_status['total_tasks']:
                batch_status['status'] = 'completed'
            elif batch_status['failed'] > 0:
                batch_status['status'] = 'partial_failure'
            elif batch_status['running'] > 0:
                batch_status['status'] = 'processing'
            else:
                batch_status['status'] = 'queued'
            
            return jsonify(batch_status)
        else:
            return jsonify({'error': 'Ошибка получения статуса пакета'}), 500
    
    except Exception as e:
        logger.error(f"Ошибка получения статуса пакета: {e}")
        return jsonify({'error': 'Ошибка получения статуса пакета'}), 500

@app.route('/api/v1/services/vllm/models', methods=['GET'])
def get_vllm_models():
    """Получение списка доступных моделей vLLM"""
    try:
        response = requests.get(f"{SERVICES['vllm']}/v1/models", timeout=10)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f'vLLM недоступен: {response.status_code}'}), 503
    except Exception as e:
        logger.error(f"Ошибка связи с vLLM: {e}")
        return jsonify({'error': f'Ошибка связи с vLLM: {e}'}), 503

@app.route('/api/v1/system/stats', methods=['GET'])
def get_system_stats():
    """Системная статистика"""
    stats = {
        'version': 'v2.0',
        'architecture': 'modular',
        'total_processed': 0,  # TODO: получать из базы данных
        'success_rate': 0.0,
        'average_processing_time': 0.0,
        'active_tasks': 0,
        'services_status': {}
    }
    
    # Получение статистики из Airflow
    try:
        response = requests.get(
            f"{SERVICES['airflow']}/api/v1/dags/orchestrator_dag/dagRuns",
            params={'limit': 100},
            headers={'Authorization': 'Basic YWRtaW46YWRtaW4='},
            timeout=10
        )
        
        if response.status_code == 200:
            dag_runs = response.json().get('dag_runs', [])
            stats['active_tasks'] = len([run for run in dag_runs if run.get('state') == 'running'])
            stats['total_processed'] = len(dag_runs)
            
            # Подсчет успешности
            successful = len([run for run in dag_runs if run.get('state') == 'success'])
            if stats['total_processed'] > 0:
                stats['success_rate'] = round(successful / stats['total_processed'] * 100, 2)
    
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
    
    # Статус каждого сервиса
    for service_name, url in SERVICES.items():
        try:
            response = requests.get(f"{url}/health", timeout=3)
            stats['services_status'][service_name] = 'healthy' if response.status_code == 200 else 'unhealthy'
        except:
            stats['services_status'][service_name] = 'unavailable'
    
    return jsonify(stats)

@app.route('/api/v1/config', methods=['GET'])
def get_config():
    """Получение конфигурации системы"""
    config = {
        'version': 'v2.0',
        'max_file_size_mb': 100,
        'supported_languages': ['original', 'ru', 'en', 'zh'],
        'quality_levels': ['basic', 'high', 'maximum'],
        'supported_formats': ['pdf'],
        'services': {name: url for name, url in SERVICES.items()},
        'features': {
            'ocr_support': True,
            'quality_assurance': True,
            'batch_processing': True,
            'multi_language': True,
            'modular_dags': True
        }
    }
    return jsonify(config)

# Создание необходимых директорий при запуске
def create_directories():
    """Создание необходимых директорий"""
    directories = [
        app.config['UPLOAD_FOLDER'],
        '/app/output',
        '/app/logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Создана директория: {directory}")

# Инициализация при запуске
create_directories()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)