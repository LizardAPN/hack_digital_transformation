import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import yaml
import os
import boto3
from io import StringIO, BytesIO
import zipfile

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_from_s3(bucket_name, file_key, endpoint_url=None, aws_access_key_id=None, aws_secret_access_key=None, region_name='ru-central1'):
    """Загрузка данных из S3-совместимого хранилища"""
    logger.info(f"Загрузка данных из s3://{bucket_name}/{file_key}")
    
    try:
        s3 = boto3.client('s3',
                          endpoint_url=endpoint_url or 'https://storage.yandexcloud.net',
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key,
                          region_name=region_name)
        
        # Загрузка данных
        s3.download_file(bucket_name, file_s3_src, file_local)
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных из S3: {str(e)}")
        raise

def load_data(file_path):
    """Загрузка данных из CSV файла (локально)"""
    logger.info(f"Загрузка данных из {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл данных не найден: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df, numeric_features, categorical_features):
    """Предобработка данных путем обработки пропущенных значений и кодирования категориальных признаков"""
    logger.info("Предобработка данных")
    
    # Обработка пропущенных значений
    for col in numeric_features:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    for col in categorical_features:
        if col in df.columns:
            df[col].fillna('Неизвестно', inplace=True)
    
    # Кодирование категориальных переменных
    df_processed = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    return df_processed

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Разделение данных на обучающую и тестовую выборки"""
    logger.info("Разделение данных на обучающую и тестовую выборки")
    
    if target_column not in df.columns:
        raise ValueError(f"Целевой столбец '{target_column}' не найден в датафрейме")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_data(X_train, X_test, y_train, y_test, train_path, test_path):
    """Сохранение обучающей и тестовой выборок в CSV файлы"""
    logger.info(f"Сохранение обучающих данных в {train_path}")
    logger.info(f"Сохранение тестовых данных в {test_path}")
    
    # Создание директорий, если они не существуют
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    # Сохранение данных
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)
    
    # Сохранение целевых переменных
    pd.DataFrame({'target': y_train}).to_csv(train_path.replace('.csv', '_target.csv'), index=False)
    pd.DataFrame({'target': y_test}).to_csv(test_path.replace('.csv', '_target.csv'), index=False)

def upload_to_s3(local_file_path, bucket_name, s3_key, endpoint_url=None, aws_access_key_id=None, aws_secret_access_key=None, region_name='ru-central1'):
    """Загрузка файла в S3-совместимое хранилище"""
    logger.info(f"Загрузка {local_file_path} в s3://{bucket_name}/{s3_key}")
    
    try:
        # Создание клиента S3
        s3 = boto3.client('s3',
                          endpoint_url=endpoint_url or f'https://storage.yandexcloud.net',
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key,
                          region_name=region_name)
        
        # Загрузка файла
        s3.upload_file(local_file_path, bucket_name, s3_key)
        logger.info(f"Файл успешно загружен в S3: s3://{bucket_name}/{s3_key}")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла в S3: {str(e)}")
        raise

def main():
    """Главная функция для подготовки набора данных"""
    # Загрузка конфигурации
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Параметры из конфигурации
    params = config
    
    # Параметры S3 из переменных окружения
    s3_bucket = os.getenv('S3_BUCKET', 's3-dvc')
    s3_raw_data_key = os.getenv('S3_RAW_DATA_KEY', 'Датасет.zip')
    s3_processed_prefix = os.getenv('S3_PROCESSED_PREFIX', 'data/processed/')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.getenv('AWS_ENDPOINT_URL', 'https://storage.yandexcloud.net')
    region_name = os.getenv('AWS_REGION', 'ru-central1')
    
    try:
        # Загрузка сырых данных из S3 или локально
        if s3_bucket and aws_access_key_id and aws_secret_access_key:
            df = load_data_from_s3(
                s3_bucket, 
                s3_raw_data_key, 
                endpoint_url, 
                aws_access_key_id, 
                aws_secret_access_key,
                region_name
            )
        else:
            raw_data_path = os.path.join(config['data']['raw_path'], 'data.csv')
            df = load_data(raw_data_path)
        
        # Предобработка данных
        df_processed = preprocess_data(
            df, 
            params['features']['numeric_features'], 
            params['features']['categorical_features']
        )
        
        # Разделение данных
        X_train, X_test, y_train, y_test = split_data(
            df_processed, 
            config['data']['target_column'],
            params['data']['test_size'],
            params['data']['random_state']
        )
        
        # Сохранение обработанных данных локально
        train_path = os.path.join(config['data']['processed_path'], config['data']['train_file'])
        test_path = os.path.join(config['data']['processed_path'], config['data']['test_file'])
        
        save_data(X_train, X_test, y_train, y_test, train_path, test_path)
        
        # Загрузка обработанных данных в S3, если указаны credentials
        if s3_bucket and aws_access_key_id and aws_secret_access_key:
            upload_to_s3(train_path, s3_bucket, s3_processed_prefix + config['data']['train_file'], 
                         endpoint_url, aws_access_key_id, aws_secret_access_key, region_name)
            upload_to_s3(test_path, s3_bucket, s3_processed_prefix + config['data']['test_file'], 
                         endpoint_url, aws_access_key_id, aws_secret_access_key, region_name)
            upload_to_s3(train_path.replace('.csv', '_target.csv'), s3_bucket, 
                         s3_processed_prefix + config['data']['train_file'].replace('.csv', '_target.csv'), 
                         endpoint_url, aws_access_key_id, aws_secret_access_key, region_name)
            upload_to_s3(test_path.replace('.csv', '_target.csv'), s3_bucket, 
                         s3_processed_prefix + config['data']['test_file'].replace('.csv', '_target.csv'), 
                         endpoint_url, aws_access_key_id, aws_secret_access_key, region_name)
        
        logger.info("Подготовка данных успешно завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в подготовке данных: {str(e)}")
        raise

if __name__ == "__main__":
    main()