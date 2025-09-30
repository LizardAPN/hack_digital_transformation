import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestFaissIndexer(unittest.TestCase):
    """Тесты для класса FaissIndexer"""

    def setUp(self):
        """Настройка тестовых данных перед каждым тестом."""
        pass

    def tearDown(self):
        """Очистка после каждого теста."""
        pass

    def test_faiss_indexer_initialization(self):
        """Тест инициализации FaissIndexer"""
        from src.data.faiss_indexer import FaissIndexer

        # Тест инициализации по умолчанию
        indexer = FaissIndexer()
        self.assertEqual(indexer.dimension, 2048)
        self.assertIsNone(indexer.index)
        self.assertEqual(indexer.image_mapping, {})

        # Тест пользовательской размерности
        indexer_custom = FaissIndexer(dimension=1024)
        self.assertEqual(indexer_custom.dimension, 1024)

    @patch("src.data.faiss_indexer.faiss")
    def test_create_index_flat(self, mock_faiss):
        """Тест создания плоского индекса FAISS"""
        from src.data.faiss_indexer import FaissIndexer

        # Мокаем классы FAISS
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_index

        # Создаем тестовый словарь признаков
        features_dict = {
            "image1.jpg": {"features": np.random.rand(2048).astype("float32"), "s3_key": "image1.jpg"},
            "image2.jpg": {"features": np.random.rand(2048).astype("float32"), "s3_key": "image2.jpg"},
        }

        # Создаем индексер и создаем индекс
        indexer = FaissIndexer()
        result = indexer.create_index(features_dict, index_type="Flat")

        # Проверяем результаты
        self.assertEqual(result, 2)
        self.assertIsNotNone(indexer.index)
        self.assertEqual(len(indexer.image_mapping), 2)
        mock_faiss.IndexFlatL2.assert_called_once_with(2048)
        mock_index.add.assert_called_once()

    @patch("src.data.faiss_indexer.faiss")
    def test_create_index_ivf(self, mock_faiss):
        """Тест создания IVF индекса FAISS"""
        from src.data.faiss_indexer import FaissIndexer

        # Мокаем классы FAISS
        mock_quantizer = Mock()
        mock_index = Mock()
        mock_faiss.IndexFlatL2.return_value = mock_quantizer
        mock_faiss.IndexIVFFlat.return_value = mock_index

        # Мокаем FAISS_CONFIG
        with patch("src.data.faiss_indexer.FAISS_CONFIG", {"nlist": 100}):
            # Создаем тестовый словарь признаков
            features_dict = {"image1.jpg": {"features": np.random.rand(2048).astype("float32"), "s3_key": "image1.jpg"}}

            # Создаем индексер и создаем индекс
            indexer = FaissIndexer()
            result = indexer.create_index(features_dict, index_type="IVF")

            # Проверяем результаты
            self.assertEqual(result, 1)
            self.assertIsNotNone(indexer.index)
            mock_faiss.IndexIVFFlat.assert_called_once()
            mock_index.train.assert_called_once()
            mock_index.add.assert_called_once()

    @patch("src.data.faiss_indexer.faiss")
    def test_search_similar(self, mock_faiss):
        """Тест поиска похожих изображений"""
        from src.data.faiss_indexer import FaissIndexer

        # Создаем индексер с замоканным индексом
        indexer = FaissIndexer()
        indexer.index = Mock()

        # Настраиваем мок отображения изображений
        indexer.image_mapping = {0: {"s3_key": "image1.jpg"}, 1: {"s3_key": "image2.jpg"}}

        # Мокаем результаты поиска
        mock_distances = np.array([[0.1, 0.5]])
        mock_indices = np.array([[0, 1]])
        indexer.index.search.return_value = (mock_distances, mock_indices)

        # Тест поиска
        query_features = np.random.rand(2048).astype("float32")
        results = indexer.search_similar(query_features, k=2)

        # Проверяем результаты
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["s3_key"], "image1.jpg")
        self.assertEqual(results[0]["distance"], 0.1)
        self.assertAlmostEqual(results[0]["similarity_score"], 1 / (1 + 0.1))
        self.assertEqual(results[1]["rank"], 2)
        self.assertEqual(results[1]["s3_key"], "image2.jpg")

        # Проверяем, что index.search был вызван правильно
        indexer.index.search.assert_called_once()

    def test_search_similar_no_index(self):
        """Тест поиска похожих изображений, когда индекс не инициализирован"""
        from src.data.faiss_indexer import FaissIndexer

        indexer = FaissIndexer()

        # Тест поиска без индекса
        query_features = np.random.rand(2048).astype("float32")

        with self.assertRaises(ValueError) as context:
            indexer.search_similar(query_features)

        self.assertIn("Индекс не инициализирован", str(context.exception))

    @patch("src.data.faiss_indexer.os.makedirs")
    @patch("src.data.faiss_indexer.faiss")
    @patch("src.data.faiss_indexer.pickle")
    def test_save_index(self, mock_pickle, mock_faiss, mock_makedirs):
        """Тест сохранения индекса и отображения FAISS"""
        from src.data.faiss_indexer import FaissIndexer

        # Создаем индексер с замоканным индексом
        indexer = FaissIndexer()
        indexer.index = Mock()
        indexer.image_mapping = {"test": "mapping"}

        # Мокаем операции с файлами
        mock_open = unittest.mock.mock_open()
        with patch("builtins.open", mock_open):
            indexer.save_index("/path/to/index.bin", "/path/to/mapping.pkl")

        # Проверяем, что моки были вызваны
        mock_makedirs.assert_called()
        mock_faiss.write_index.assert_called_once_with(indexer.index, "/path/to/index.bin")
        mock_open.assert_called_once_with("/path/to/mapping.pkl", "wb")
        mock_pickle.dump.assert_called_once()

    @patch("src.data.faiss_indexer.faiss")
    @patch("src.data.faiss_indexer.pickle")
    def test_load_index(self, mock_pickle, mock_faiss):
        """Тест загрузки индекса и отображения FAISS"""
        from src.data.faiss_indexer import FaissIndexer

        # Мокаем операции с файлами
        mock_index = Mock()
        mock_faiss.read_index.return_value = mock_index

        mock_open = unittest.mock.mock_open()
        with patch("builtins.open", mock_open):
            mock_pickle.load.return_value = {"test": "mapping"}

            # Создаем индексер и загружаем индекс
            indexer = FaissIndexer()
            indexer.load_index("/path/to/index.bin", "/path/to/mapping.pkl")

        # Проверяем результаты
        self.assertEqual(indexer.index, mock_index)
        self.assertEqual(indexer.image_mapping, {"test": "mapping"})
        mock_faiss.read_index.assert_called_once_with("/path/to/index.bin")
        mock_open.assert_called_once_with("/path/to/mapping.pkl", "rb")
        mock_pickle.load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
