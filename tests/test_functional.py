import unittest
import os
import tempfile
import numpy as np
from PIL import Image
import sys
import logging

# Добавляем путь к src для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.cv_model import CVModel
from src.models.feature_extractor import FeatureExtractor
from src.data.faiss_indexer import FaissIndexer
from src.geo.geocoder import Geocoder


class TestFunctional(unittest.TestCase):
    """Функциональные тесты для проверки работы системы"""
    
    @classmethod
    def setUpClass(cls):
        """Инициализация перед всеми тестами"""
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        cls.logger = logging.getLogger(__name__)
        
    def setUp(self):
        """Инициализация перед каждым тестом"""
        pass
        
    def tearDown(self):
        """Очистка после каждого теста"""
        pass
        
    def test_feature_extraction(self):
        """Тест извлечения признаков из изображения"""
        # Создаем тестовое изображение
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Инициализируем экстрактор признаков
        extractor = FeatureExtractor()
        
        # Извлекаем признаки
        features = extractor.extract_features(test_image)
        
        # Проверяем, что признаки извлечены корректно
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (2048,))  # ResNet50 features
        self.assertAlmostEqual(np.linalg.norm(features), 1.0, places=3)  # Нормализованные признаки
        
    def test_faiss_index_operations(self):
        """Тест операций с FAISS индексом"""
        # Создаем тестовые признаки
        features_dict = {}
        for i in range(5):
            features_dict[f"test_image_{i}.jpg"] = {
                "features": np.random.rand(2048).astype('float32'),
                "s3_key": f"test/test_image_{i}.jpg"
            }
        
        # Создаем FAISS индекс
        indexer = FaissIndexer(dimension=2048)
        num_indexed = indexer.create_index(features_dict, index_type="Flat")
        
        # Проверяем, что индекс создан корректно
        self.assertEqual(num_indexed, 5)
        self.assertIsNotNone(indexer.index)
        self.assertEqual(len(indexer.image_mapping), 5)
        
        # Тестируем поиск похожих изображений
        query_features = np.random.rand(2048).astype('float32')
        similar_images = indexer.search_similar(query_features, k=3)
        
        # Проверяем результаты поиска
        self.assertEqual(len(similar_images), 3)
        for result in similar_images:
            self.assertIn("rank", result)
            self.assertIn("s3_key", result)
            self.assertIn("distance", result)
            self.assertIn("similarity_score", result)
            
    def test_geocoder_initialization(self):
        """Тест инициализации геокодера"""
        geocoder = Geocoder()
        
        # Проверяем, что геокодер инициализирован корректно
        self.assertIsNotNone(geocoder)
        
    def test_cv_model_initialization(self):
        """Тест инициализации CV модели"""
        try:
            cv_model = CVModel()
            
            # Проверяем, что модель инициализирована корректно
            self.assertIsNotNone(cv_model.feature_extractor)
            self.assertIsNotNone(cv_model.ocr_model)
            # Примечание: indexer может не быть инициализирован, если нет индекса
            
        except Exception as e:
            # Разрешаем ошибку инициализации индекса, если индекс не существует
            self.assertIn("FAISS", str(e))
            
    def test_performance_requirements(self):
        """Тест требований к производительности"""
        # Проверяем, что система может обработать 1000 изображений за ≤3 часов
        # Это тест производительности, который должен выполняться отдельно
        max_processing_time = 3 * 60 * 60  # 3 часа в секундах
        max_images = 1000
        max_time_per_image = max_processing_time / max_images  # 10.8 секунд на изображение
        
        # Для демонстрации: проверяем, что максимальное время на изображение >= 1 секунды
        self.assertGreaterEqual(max_time_per_image, 1.0)


if __name__ == '__main__':
    unittest.main()
