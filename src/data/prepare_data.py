import sys
from pathlib import Path

# Добавляем путь к утилитам для корректной работы импортов
utils_path = Path(__file__).resolve().parent.parent / "utils"
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

# Настраиваем пути проекта
try:
    from path_resolver import setup_project_paths
    setup_project_paths()
except ImportError:
    # Если path_resolver недоступен, добавляем необходимые пути вручную
    src_path = Path(__file__).resolve().parent.parent
    paths_to_add = [src_path, src_path / "utils", src_path / "geo"]
    for path in paths_to_add:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class PrepareData(Dataset):
    """
    Класс для подготовки датасета из изображений и их географических координат.
    Автоматически разделяет данные на тренировочную и тестовую выборки с разными трансформациями.

    Attributes
    ----------
    images_dir : str
        Путь к директории с изображениями.
    transform : callable or None
        Трансформации для применения к изображениям.
    df : pandas.DataFrame
        Данные из CSV-файла.
    train_indices : list
        Индексы тренировочной выборки.
    test_indices : list
        Индексы тестовой выборки.
    """

    def __init__(self, csv_path, images_dir, test_size=0.2, random_state=42):
        """
        Инициализация датасета.

        Parameters
        ----------
        csv_path : str or Path
            Путь к CSV-файлу с данными.
        images_dir : str or Path
            Путь к директории с изображениями.
        test_size : float, optional
            Доля тестовой выборки, by default 0.2
        random_state : int, optional
            Random state для воспроизводимости, by default 42
        """
        self.images_dir = Path(images_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # Загрузка данных из CSV
        self.df = pd.read_csv(csv_path)
        print(f"Загружено {len(self.df)} записей из CSV-файла")

        # Фильтрация данных по наличию изображений
        self._filter_by_images()
        print(f"После фильтрации по изображениям осталось {len(self.df)} записей")

        # Разделение данных
        if len(self.df) > 0:
            self._split_data(test_size, random_state)
        else:
            raise ValueError("После фильтрации не осталось ни одной записи")

    def _filter_by_images(self):
        """
        Фильтрация данных по наличию изображений.
        """
        # Проверяем, что файлы изображений существуют
        image_files = set(
            f.name for f in self.images_dir.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )
        self.df = self.df[self.df["camera_id"].isin(image_files)]
        self.df = self.df.reset_index(drop=True)

    def _split_data(self, test_size, random_state):
        """
        Разделение данных на тренировочную и тестовую выборки.

        Parameters
        ----------
        test_size : float
            Доля тестовой выборки.
        random_state : int
            Random state для воспроизводимости.
        """
        if len(self.df) == 0:
            self.train_indices = []
            self.test_indices = []
            return

        # Разделение индексов
        indices = list(range(len(self.df)))
        self.train_indices, self.test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        print(f"Разделение данных: {len(self.train_indices)} тренировочных, {len(self.test_indices)} тестовых")

    def get_train_dataset(self):
        """
        Получение тренировочного датасета.

        Returns
        -------
        PrepareData
            Тренировочный датасет.
        """
        return self._create_subset(self.train_indices)

    def get_test_dataset(self):
        """
        Получение тестового датасета.

        Returns
        -------
        PrepareData
            Тестовый датасет.
        """
        return self._create_subset(self.test_indices)

    def _create_subset(self, indices):
        """
        Создание подмножества датасета по индексам.

        Parameters
        ----------
        indices : list
            Список индексов.

        Returns
        -------
        PrepareData
            Подмножество датасета.
        """
        # Создаем копию объекта
        subset = PrepareData.__new__(PrepareData)
        subset.images_dir = self.images_dir
        subset.transform = self.transform
        subset.df = self.df.iloc[indices].reset_index(drop=True)
        # Для подмножества не нужно разделять данные
        subset.train_indices = list(range(len(subset.df)))
        subset.test_indices = []
        return subset

    def __len__(self):
        """
        Получение длины датасета.

        Returns
        -------
        int
            Длина датасета.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Получение элемента датасета по индексу.

        Parameters
        ----------
        idx : int
            Индекс элемента.

        Returns
        -------
        tuple
            Кортеж из изображения и координат (тензор изображения, тензор координат).
        """
        row = self.df.iloc[idx]
        image_path = self.images_dir / row["camera_id"]

        # Загрузка изображения
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Получение координат
        coordinates = torch.tensor([row["lat_real"], row["lon_real"]], dtype=torch.float32)

        return image, coordinates
