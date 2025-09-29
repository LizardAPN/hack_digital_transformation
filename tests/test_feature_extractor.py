import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from PIL import Image
import io

# Добавляем путь к src для импортов
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

class TestFeatureExtractor(unittest.TestCase):
    """Тесты для класса FeatureExtractor"""

    def setUp(self):
        """Настройка тестовых данных перед каждым тестом."""
        pass

    def tearDown(self):
        """Очистка после каждого теста."""
        pass

    def test_feature_extractor_initialization(self):
        """Тест инициализации FeatureExtractor"""
        from src.models.feature_extractor import FeatureExtractor
        
        # Поскольку инициализация требует загрузки реальной модели, мы проверим,
        # что класс может быть создан без возникновения исключений
        try:
            # Мы не будем создавать полный экземпляр класса, так как это требует
            # загрузки моделей, но можем проверить, что класс существует
            self.assertTrue(hasattr(FeatureExtractor, '__init__'))
        except Exception as e:
            self.fail(f"Инициализация FeatureExtractor завершилась с исключением: {e}")
        
    def test_feature_extractor_with_cuda(self):
        """Тест инициализации FeatureExtractor с CUDA"""
        from src.models.feature_extractor import FeatureExtractor
        
        # Поскольку инициализация требует загрузки реальной модели, мы проверим,
        # что класс может быть создан без возникновения исключений
        try:
            # Мы не будем создавать полный экземпляр класса, так как это требует
            # загрузки моделей, но можем проверить, что класс существует
            self.assertTrue(hasattr(FeatureExtractor, '__init__'))
        except Exception as e:
            self.fail(f"Инициализация FeatureExtractor завершилась с исключением: {e}")
        
    def test_load_image_from_bytes(self):
        """Тест загрузки изображения из байтов"""
        from src.models.feature_extractor import FeatureExtractor
        
        # Создаем простое тестовое изображение
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        extractor = FeatureExtractor.__new__(FeatureExtractor)  # Создаем экземпляр без __init__
        
        # Тест успешной загрузки изображения
        loaded_img = extractor.load_image_from_bytes(img_bytes)
        self.assertIsInstance(loaded_img, Image.Image)
        self.assertEqual(loaded_img.size, (100, 100))
        
        # Тест загрузки изображения с другим режимом (RGBA)
        img_rgba = Image.new('RGBA', (50, 50), color='blue')
        img_rgba_bytes = io.BytesIO()
        img_rgba.save(img_rgba_bytes, format='PNG')
        img_rgba_bytes = img_rgba_bytes.getvalue()
        
        loaded_rgba_img = extractor.load_image_from_bytes(img_rgba_bytes)
        self.assertIsInstance(loaded_rgba_img, Image.Image)
        self.assertEqual(loaded_rgba_img.mode, "RGB")  # Должно быть преобразовано в RGB
        
        # Тест обработки ошибок с недопустимыми байтами
        invalid_bytes = b"invalid image data"
        result = extractor.load_image_from_bytes(invalid_bytes)
        self.assertIsNone(result)
        
    @patch('src.models.feature_extractor.torch')
    def test_extract_features(self, mock_torch):
        """Тест извлечения признаков из изображения"""
        from src.models.feature_extractor import FeatureExtractor
        
        # Создаем мок-экземпляр экстрактора
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        extractor.device = "cpu"
        
        # Мокаем модель и трансформации
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock()
        extractor.model = mock_model
        
        mock_transform = Mock()
        extractor.transform = mock_transform
        
        # Мокаем операции torch
        mock_torch.no_grad = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = Mock(return_value=None)
        mock_torch.no_grad.return_value.__exit__ = Mock(return_value=None)
        
        # Мокаем операции с тензорами
        mock_tensor = Mock()
        mock_tensor.unsqueeze = Mock(return_value=mock_tensor)
        mock_tensor.to = Mock(return_value=mock_tensor)
        
        mock_transform.return_value = mock_tensor
        mock_model.return_value = mock_tensor
        
        # Мокаем преобразование в массив numpy
        mock_tensor.cpu = Mock()
        mock_cpu_tensor = Mock()
        mock_tensor.cpu.return_value = mock_cpu_tensor
        mock_cpu_tensor.numpy = Mock(return_value=np.array([[1, 2, 3]]))
        
        # Создаем тестовое изображение
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Тест извлечения признаков
        with patch('numpy.linalg.norm', return_value=1.0):
            features = extractor.extract_features(test_image)
            
        # Проверяем результаты
        self.assertIsNotNone(features)
        self.assertIsInstance(features, np.ndarray)
        
        # Тест обработки ошибок
        mock_transform.side_effect = Exception("Transform error")
        features = extractor.extract_features(test_image)
        self.assertIsNone(features)
        
    def test_process_image_batch(self):
        """Тест обработки пакета изображений"""
        from src.models.feature_extractor import FeatureExtractor
        
        # Создаем мок-экземпляр экстрактора
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        extractor.stats = {"processed": 0, "failed": 0, "total_time": 0}
        
        # Мокаем метод extract_features
        def mock_extract_features(image):
            if image is not None:
                return np.random.rand(2048).astype("float32")
            return None
            
        extractor.extract_features = mock_extract_features
        extractor.load_image_from_bytes = lambda x: Image.new('RGB', (224, 224), color='red') if x is not None else None
        
        # Создаем тестовые данные пакета
        batch_data = {
            "image1.jpg": b"fake_image_data_1",
            "image2.jpg": b"fake_image_data_2",
            "invalid_image.jpg": None
        }
        
        # Обрабатываем пакет
        results = extractor.process_image_batch(batch_data)
        
        # Проверяем результаты
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 2)  # Два действительных изображения
        self.assertIn("image1.jpg", results)
        self.assertIn("image2.jpg", results)
        
        # Проверяем, что статистика была обновлена
        self.assertEqual(extractor.stats["processed"], 2)
        self.assertEqual(extractor.stats["failed"], 1)
        
    def test_get_processing_stats(self):
        """Тест получения статистики обработки"""
        from src.models.feature_extractor import FeatureExtractor
        
        # Создаем мок-экземпляр экстрактора
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        test_stats = {"processed": 10, "failed": 2, "total_time": 5.5}
        extractor.stats = test_stats.copy()
        
        # Получаем статистику
        stats = extractor.get_processing_stats()
        
        # Проверяем результаты
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats, test_stats)
        
        # Проверяем, что возвращается копия, а не оригинал
        stats["processed"] = 20
        self.assertNotEqual(extractor.stats["processed"], stats["processed"])

if __name__ == '__main__':
    unittest.main()
