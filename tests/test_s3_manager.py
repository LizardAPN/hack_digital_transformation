import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Добавляем путь к src для импортов
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestS3Manager(unittest.TestCase):
    """Тесты для класса S3Manager"""

    def setUp(self):
        """Настройка тестовых данных перед каждым тестом."""
        pass

    def tearDown(self):
        """Очистка после каждого теста."""
        pass

    @patch("src.utils.s3_optimize.boto3")
    def test_s3_manager_initialization(self, mock_boto3):
        """Тест инициализации S3Manager"""
        from src.utils.s3_optimize import S3Manager

        # Мокаем переменные окружения
        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret", "AWS_BUCKET_NAME": "test_bucket"},
        ):
            # Мокаем клиент boto3
            mock_client = Mock()
            mock_boto3.client.return_value = mock_client
            mock_client.head_bucket.return_value = True

            # Создаем экземпляр S3Manager
            s3_manager = S3Manager()

            # Проверяем инициализацию
            self.assertEqual(s3_manager.key_id, "test_key")
            self.assertEqual(s3_manager.access_key, "test_secret")
            self.assertEqual(s3_manager.bucket_name, "test_bucket")
            self.assertEqual(s3_manager.max_workers, 10)
            self.assertEqual(s3_manager.chunk_size, 8 * 1024 * 1024)

            # Проверяем создание клиента
            mock_boto3.client.assert_called_once_with(
                "s3",
                aws_access_key_id="test_key",
                aws_secret_access_key="test_secret",
                endpoint_url="https://s3-msk.tinkoff.ru",
            )

    @patch("src.utils.s3_optimize.boto3")
    def test_s3_manager_initialization_with_params(self, mock_boto3):
        """Тест инициализации S3Manager с пользовательскими параметрами"""
        from src.utils.s3_optimize import S3Manager

        # Мокаем клиент boto3
        mock_client = Mock()
        mock_boto3.client.return_value = mock_client
        mock_client.head_bucket.return_value = True

        # Создаем экземпляр S3Manager с пользовательскими параметрами
        s3_manager = S3Manager(
            key_id="custom_key",
            access_key="custom_secret",
            endpoint_url="https://custom.endpoint.com",
            bucket_name="custom_bucket",
            max_workers=20,
            chunk_size=4 * 1024 * 1024,
        )

        # Проверяем инициализацию
        self.assertEqual(s3_manager.key_id, "custom_key")
        self.assertEqual(s3_manager.access_key, "custom_secret")
        self.assertEqual(s3_manager.endpoint_url, "https://custom.endpoint.com")
        self.assertEqual(s3_manager.bucket_name, "custom_bucket")
        self.assertEqual(s3_manager.max_workers, 20)
        self.assertEqual(s3_manager.chunk_size, 4 * 1024 * 1024)

    @patch("src.utils.s3_optimize.boto3")
    def test_s3_manager_missing_credentials(self, mock_boto3):
        """Тест инициализации S3Manager с отсутствующими учетными данными"""
        from src.utils.s3_optimize import S3Manager

        # Тест без переменных окружения
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                S3Manager()

            self.assertIn("Missing required AWS credentials", str(context.exception))

    @patch("src.utils.s3_optimize.boto3")
    def test_upload_bytes(self, mock_boto3):
        """Тест загрузки байтов в S3"""
        from src.utils.s3_optimize import S3Manager

        # Мокаем переменные окружения
        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret", "AWS_BUCKET_NAME": "test_bucket"},
        ):
            # Мокаем клиент boto3
            mock_client = Mock()
            mock_boto3.client.return_value = mock_client
            mock_client.head_bucket.return_value = True

            # Создаем экземпляр S3Manager
            s3_manager = S3Manager()

            # Тест upload_bytes
            test_data = b"test data"
            result = s3_manager.upload_bytes(test_data, "test_key")

            # Проверяем результаты
            self.assertTrue(result)
            mock_client.upload_fileobj.assert_called_once()

            # Проверяем, что статистика была обновлена
            self.assertEqual(s3_manager._upload_stats["successful"], 1)
            self.assertEqual(s3_manager._upload_stats["total_size"], len(test_data))

    @patch("src.utils.s3_optimize.boto3")
    def test_download_bytes(self, mock_boto3):
        """Тест скачивания байтов из S3"""
        from src.utils.s3_optimize import S3Manager

        # Мокаем переменные окружения
        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret", "AWS_BUCKET_NAME": "test_bucket"},
        ):
            # Мокаем клиент boto3
            mock_client = Mock()
            mock_boto3.client.return_value = mock_client
            mock_client.head_bucket.return_value = True

            # Мокаем ответ загрузки
            mock_body = Mock()
            mock_body.read.return_value = b"test downloaded data"
            mock_client.download_fileobj.return_value = None

            # Создаем экземпляр S3Manager
            s3_manager = S3Manager()

            # Тест download_bytes
            result = s3_manager.download_bytes("test_key")

            # Проверяем результаты
            self.assertIsNotNone(result)
            self.assertIsInstance(result, bytes)
            mock_client.download_fileobj.assert_called_once()

            # Проверяем, что статистика была обновлена
            self.assertEqual(s3_manager._download_stats["successful"], 1)
            self.assertEqual(s3_manager._download_stats["total_size"], len(result))

    @patch("src.utils.s3_optimize.boto3")
    def test_list_files(self, mock_boto3):
        """Тест получения списка файлов в S3"""
        from src.utils.s3_optimize import S3Manager

        # Мокаем переменные окружения
        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret", "AWS_BUCKET_NAME": "test_bucket"},
        ):
            # Мокаем клиент boto3
            mock_client = Mock()
            mock_boto3.client.return_value = mock_client
            mock_client.head_bucket.return_value = True

            # Мокаем ответ list_objects_v2
            mock_client.list_objects_v2.return_value = {
                "Contents": [{"Key": "file1.jpg"}, {"Key": "file2.png"}, {"Key": "file3.txt"}],
                "IsTruncated": False,
            }

            # Создаем экземпляр S3Manager
            s3_manager = S3Manager()

            # Тест list_files
            result = s3_manager.list_files()

            # Проверяем результаты
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3)
            self.assertIn("file1.jpg", result)
            self.assertIn("file2.png", result)
            self.assertIn("file3.txt", result)

    @patch("src.utils.s3_optimize.boto3")
    def test_list_files_with_filtering(self, mock_boto3):
        """Тест получения списка файлов в S3 с фильтрацией по расширениям"""
        from src.utils.s3_optimize import S3Manager

        # Мокаем переменные окружения
        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret", "AWS_BUCKET_NAME": "test_bucket"},
        ):
            # Мокаем клиент boto3
            mock_client = Mock()
            mock_boto3.client.return_value = mock_client
            mock_client.head_bucket.return_value = True

            # Мокаем ответ list_objects_v2
            mock_client.list_objects_v2.return_value = {
                "Contents": [
                    {"Key": "image1.jpg"},
                    {"Key": "image2.png"},
                    {"Key": "document.pdf"},
                    {"Key": "data.csv"},
                ],
                "IsTruncated": False,
            }

            # Создаем экземпляр S3Manager
            s3_manager = S3Manager()

            # Тест list_files с фильтрацией по расширениям
            result = s3_manager.list_files(file_extensions=[".jpg", ".png"])

            # Проверяем результаты
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertIn("image1.jpg", result)
            self.assertIn("image2.png", result)
            self.assertNotIn("document.pdf", result)

    def test_get_upload_stats(self):
        """Тест получения статистики загрузки"""
        from src.utils.s3_optimize import S3Manager

        # Создаем мок-экземпляр S3Manager
        s3_manager = S3Manager.__new__(S3Manager)
        test_stats = {"successful": 5, "failed": 1, "total_size": 10240}
        s3_manager._upload_stats = test_stats.copy()

        # Получаем статистику
        stats = s3_manager.get_upload_stats()

        # Проверяем результаты
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats, test_stats)

        # Проверяем, что возвращается копия, а не оригинал
        stats["successful"] = 10
        self.assertNotEqual(s3_manager._upload_stats["successful"], stats["successful"])

    def test_get_download_stats(self):
        """Тест получения статистики скачивания"""
        from src.utils.s3_optimize import S3Manager

        # Создаем мок-экземпляр S3Manager
        s3_manager = S3Manager.__new__(S3Manager)
        test_stats = {"successful": 3, "failed": 0, "total_size": 20480}
        s3_manager._download_stats = test_stats.copy()

        # Получаем статистику
        stats = s3_manager.get_download_stats()

        # Проверяем результаты
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats, test_stats)

        # Проверяем, что возвращается копия, а не оригинал
        stats["successful"] = 7
        self.assertNotEqual(s3_manager._download_stats["successful"], stats["successful"])


if __name__ == "__main__":
    unittest.main()
