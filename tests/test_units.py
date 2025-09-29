import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Добавляем путь к src для импортов
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

class TestUnits(unittest.TestCase):
    """Модульные тесты для различных модулей проекта"""

    def test_imports(self):
        """Тест импорта всех модулей без ошибок"""
        try:
            from src.data.database_builder import create_directories, validate_s3_connection
            from src.data.faiss_indexer import FaissIndexer
            from src.models.feature_extractor import FeatureExtractor
            from src.geo.download_images import BBoxSplitter, MapillaryS3Client
            from src.utils.s3_optimize import S3Manager
            from src.utils.config import MODEL_CONFIG, FAISS_CONFIG, DATA_PATHS, PERFORMANCE_CONFIG
            from src.models.cv_model import CVModel
            from src.geo.geocoder import Geocoder
            from src.models.OCR_model import OverlayOCR
            from src.utils.monitor_database import monitor_database
            
            # Если мы дошли до этой точки, значит все импорты прошли успешно
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Ошибка импорта: {e}")

    def test_model_config(self):
        """Тест наличия обязательных ключей в MODEL_CONFIG"""
        from src.utils.config import MODEL_CONFIG
        
        required_keys = ["model_name", "input_size", "pooling"]
        for key in required_keys:
            self.assertIn(key, MODEL_CONFIG)
            
    def test_faiss_config(self):
        """Тест наличия обязательных ключей в FAISS_CONFIG"""
        from src.utils.config import FAISS_CONFIG
        
        required_keys = ["index_type", "nlist", "metric"]
        for key in required_keys:
            self.assertIn(key, FAISS_CONFIG)
            
    def test_data_paths(self):
        """Тест наличия обязательных ключей в DATA_PATHS"""
        from src.utils.config import DATA_PATHS
        
        required_keys = ["faiss_index", "mapping_file", "metadata_file"]
        for key in required_keys:
            self.assertIn(key, DATA_PATHS)

    def test_performance_config(self):
        """Тест наличия обязательных ключей в PERFORMANCE_CONFIG"""
        from src.utils.config import PERFORMANCE_CONFIG
        
        required_keys = ["max_concurrent_tasks", "batch_size", "max_image_size", 
                        "cache_features", "enable_gpu", "processing_timeout"]
        for key in required_keys:
            self.assertIn(key, PERFORMANCE_CONFIG)

    @patch('src.utils.s3_optimize.boto3')
    def test_s3_manager_init(self, mock_boto3):
        """Тест инициализации S3Manager"""
        from src.utils.s3_optimize import S3Manager
        
        # Мокаем переменные окружения
        with patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret',
            'AWS_BUCKET_NAME': 'test_bucket'
        }):
            s3_manager = S3Manager()
            self.assertIsNotNone(s3_manager)
            
    def test_bbox_splitter_static_methods(self):
        """Тест статических методов BBoxSplitter"""
        from src.geo.download_images import BBoxSplitter
        
        # Тест split_bbox
        bbox = [0, 0, 10, 10]
        sub_bboxes = BBoxSplitter.split_bbox(bbox, grid_size=2)
        self.assertEqual(len(sub_bboxes), 4)
        
        # Тест create_bbox_grid
        bboxes = BBoxSplitter.create_bbox_grid(55.7558, 37.6176, grid_radius=1, bbox_size=0.01)
        self.assertEqual(len(bboxes), 9)  # сетка 3x3
        
        # Тест calculate_optimal_grid_size
        grid_size = BBoxSplitter.calculate_optimal_grid_size(bbox, target_bbox_area=0.01)
        self.assertIsInstance(grid_size, int)
        self.assertGreaterEqual(grid_size, 1)
        
    @patch('src.geo.geocoder.requests')
    def test_geocoder_yandex_geocode(self, mock_requests):
        """Тест функциональности геокодера Yandex"""
        from src.geo.geocoder import Geocoder
        
        # Мокаем ответ
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
        
        result = geocoder._yandex_geocode(55.7558, 37.6176)
        self.assertEqual(result, "Moscow, Russia")
        
    def test_overlay_ocr_clean_token(self):
        """Тест метода _clean_token класса OverlayOCR"""
        from src.models.OCR_model import OverlayOCR
        
        ocr = OverlayOCR()
        result = ocr._clean_token("Test123!@#")
        self.assertEqual(result, "Test123")
        
    def test_overlay_ocr_alnum_class(self):
        """Тест метода _alnum_class класса OverlayOCR"""
        from src.models.OCR_model import OverlayOCR
        
        ocr = OverlayOCR()
        self.assertEqual(ocr._alnum_class("a"), "A")
        self.assertEqual(ocr._alnum_class("1"), "D")
        self.assertEqual(ocr._alnum_class("!"), "_")

if __name__ == '__main__':
    unittest.main()
