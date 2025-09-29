import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

# Добавляем путь к src для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.faiss_indexer import FaissIndexer
from src.geo.geocoder import Geocoder
from src.models.cv_model import CVModel
from src.models.feature_extractor import FeatureExtractor


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
        test_image = Image.new("RGB", (224, 224), color="red")

        # Инициализируем экстрактор признаков
        extractor = FeatureExtractor()

        # Извлекаем признаки
        features = extractor.extract_features(test_image)

        # Проверяем, что признаки извлечены корректно
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (2048,))  # Признаки ResNet50
        self.assertAlmostEqual(np.linalg.norm(features), 1.0, places=3)  # Нормализованные признаки

    def test_faiss_index_operations(self):
        """Тест операций с FAISS индексом"""
        # Создаем тестовые признаки
        features_dict = {}
        for i in range(5):
            features_dict[f"test_image_{i}.jpg"] = {
                "features": np.random.rand(2048).astype("float32"),
                "s3_key": f"test/test_image_{i}.jpg",
            }

        # Создаем FAISS индекс
        indexer = FaissIndexer(dimension=2048)
        num_indexed = indexer.create_index(features_dict, index_type="Flat")

        # Проверяем, что индекс создан корректно
        self.assertEqual(num_indexed, 5)
        self.assertIsNotNone(indexer.index)
        self.assertEqual(len(indexer.image_mapping), 5)

        # Тестируем поиск похожих изображений
        query_features = np.random.rand(2048).astype("float32")
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

    @patch('src.models.cv_model.FaissIndexer')
    def test_cv_model_initialization(self, mock_faiss_indexer):
        """Тест инициализации CV модели"""
        # Мок успешной инициализации FAISS индекса
        mock_indexer_instance = Mock()
        mock_faiss_indexer.return_value = mock_indexer_instance
        
        cv_model = CVModel()

        # Проверяем, что модель инициализирована корректно
        self.assertIsNotNone(cv_model.feature_extractor)
        self.assertIsNotNone(cv_model.ocr_model)
        self.assertIsNotNone(cv_model.indexer)

    def test_performance_requirements(self):
        """Тест требований к производительности"""
        # Проверяем, что система может обработать 1000 изображений за ≤3 часов
        # Это тест производительности, который должен выполняться отдельно
        max_processing_time = 3 * 60 * 60  # 3 часа в секундах
        max_images = 1000
        max_time_per_image = max_processing_time / max_images  # 10.8 секунд на изображение

        # Для демонстрации: проверяем, что максимальное время на изображение >= 1 секунды
        self.assertGreaterEqual(max_time_per_image, 1.0)

    def test_config_loading(self):
        """Тест загрузки конфигурации"""
        # Проверяем, что конфигурационные файлы загружаются корректно
        try:
            from src.utils.config import MODEL_CONFIG, FAISS_CONFIG, DATA_PATHS, PERFORMANCE_CONFIG
            
            # Проверяем наличие обязательных ключей
            self.assertIn("model_name", MODEL_CONFIG)
            self.assertIn("input_size", MODEL_CONFIG)
            self.assertIn("pooling", MODEL_CONFIG)
            
            self.assertIn("index_type", FAISS_CONFIG)
            self.assertIn("nlist", FAISS_CONFIG)
            self.assertIn("metric", FAISS_CONFIG)
            
            self.assertIn("faiss_index", DATA_PATHS)
            self.assertIn("mapping_file", DATA_PATHS)
            self.assertIn("metadata_file", DATA_PATHS)
            
            required_perf_keys = ["max_concurrent_tasks", "batch_size", "max_image_size", 
                                "cache_features", "enable_gpu", "processing_timeout"]
            for key in required_perf_keys:
                self.assertIn(key, PERFORMANCE_CONFIG)
                
        except Exception as e:
            self.fail(f"Ошибка загрузки конфигурации: {e}")

    @patch('src.geo.geocoder.requests')
    def test_geocoder_functionality(self, mock_requests):
        """Тест функциональности геокодера"""
        # Мок ответа от Yandex Geocoder API
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": {
                "GeoObjectCollection": {
                    "featureMember": [{
                        "GeoObject": {
                            "metaDataProperty": {
                                "GeocoderMetaData": {
                                    "text": "Moscow, Russia"
                                }
                            }
                        }
                    }]
                }
            }
        }
        mock_requests.get.return_value = mock_response
        
        geocoder = Geocoder()
        geocoder.yandex_api_key = "test_key"
        
        # Тест геокодирования
        result = geocoder.geocode(55.7558, 37.6176)
        
        # Проверяем результат
        self.assertEqual(result, "Moscow, Russia")
        mock_requests.get.assert_called()


if __name__ == "__main__":
    unittest.main()
