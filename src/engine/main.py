import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ ДЛЯ DATASPHERE
# =============================================================================


# Динамическое определение путей для работы в DataSphere
current_file = Path(__file__).resolve()
project_root_in_cloud = Path("/job")  # Явно указываем корень в облаке
local_project_root = current_file.parent.parent

# Выбираем корень в зависимости от окружения
# Проверяем, находимся ли мы в среде DataSphere (существует ли папка /job)
if project_root_in_cloud.exists():
    ROOT_DIR = project_root_in_cloud
    print("✓ Обнаружена среда DataSphere. Используем путь /job")
else:
    ROOT_DIR = local_project_root
    print("✓ Обнаружена локальная среда. Используем локальный путь")

# Добавляем возможные пути к модулям в sys.path
possible_paths_to_models = [
    ROOT_DIR / "models",  # Папка models в корне
    ROOT_DIR / "src" / "models",  # Папка models внутри src
    ROOT_DIR,  # Сам корень проекта
    ROOT_DIR / "src",  # Папка src
    ROOT_DIR / "utils",  # Папка utils в корне
    ROOT_DIR / "src" / "utils",  # Папка utils внутри src
]

for path in possible_paths_to_models:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)
        print(f"✓ Добавлен путь: {path}")

# Также добавляем родительскую директорию текущего файла
current_parent = str(current_file.parent)
if current_parent not in sys.path:
    sys.path.insert(0, current_parent)

print("=" * 60)
print("FINAL ENVIRONMENT INFO:")
print(f"Current file: {current_file}")
print(f"ROOT_DIR: {ROOT_DIR}")
print(f"Current working directory: {Path.cwd()}")
print(f"Python will look for modules in:")
for i, path in enumerate(sys.path[:10]):  # Показываем первые 10 путей
    print(f"  {i+1}. {path}")
print("=" * 60)

# Диагностика: что действительно есть в облаке
print("\nCHECKING CLOUD ENVIRONMENT STRUCTURE:")
check_paths = [ROOT_DIR, Path(".")]
for path in check_paths:
    if path.exists():
        print(f"\nСодержимое {path}:")
        try:
            items = list(path.iterdir())
            if not items:
                print("  [EMPTY]")
            for item in items:
                item_type = "DIR" if item.is_dir() else "FILE"
                print(f"  [{item_type}] {item.name}")
        except Exception as e:
            print(f"  Ошибка доступа: {e}")
print("=" * 60)

# Теперь пробуем импортировать
try:
    from models.OCR_model import OverlayOCR

    print("✓ Модуль models.OCR_model успешно импортирован")
except ImportError as e:
    print(f"✗ Ошибка импорта models.OCR_model: {e}")
    # Попробуем альтернативный путь
    try:
        # Если модуль в той же директории, что и main.py
        from OCR_model import OverlayOCR

        print("✓ Модуль OCR_model успешно импортирован из текущей директории")
    except ImportError as e2:
        print(f"✗ Ошибка импорта OCR_model: {e2}")
        raise

try:
    from utils.useful_functions import levenshtein_distance

    print("✓ Модуль utils.useful_functions успешно импортирован")
except ImportError as e:
    print(f"✗ Ошибка импорта utils.useful_functions: {e}")
    # Попробуем альтернативный путь
    try:
        from useful_functions import levenshtein_distance

        print("✓ Модуль useful_functions успешно импортирован из текущей директории")
    except ImportError as e2:
        print(f"✗ Ошибка импорта useful_functions: {e2}")

        # Создаем заглушку, если функция не найдена
        def levenshtein_distance(s1, s2):
            print(f"WARNING: Using dummy levenshtein_distance for '{s1}' and '{s2}'")
            return abs(len(s1) - len(s2))

        print("✓ Создана заглушка для levenshtein_distance")

warnings.filterwarnings("ignore")

# Директория для сохранения результатов
save_dir = ROOT_DIR / "models" / "ocr_model"
save_dir.mkdir(parents=True, exist_ok=True)
print(f"Save directory: {save_dir}")


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="OCR Model Training")
    parser.add_argument(
        "--csv-path", type=str, default="data/processed_data/merged_data.csv", help="Путь к CSV файлу с данными"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/raw_data/data/metadata/INC/united_image/",
        help="Директория с изображениями",
    )
    parser.add_argument("--optuna-study", type=str, default="optuna_study.pkl", help="Имя файла для сохранения study")
    parser.add_argument(
        "--ocr-model-params", type=str, default="ocr_model_params.json", help="Имя файла для сохранения параметров"
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Количество trials для Optuna")
    parser.add_argument("--max-samples", type=int, default=200, help="Максимальное количество образцов для оценки")
    return parser.parse_args()


class OCRDataset(Dataset):
    """Датасет для обучения OCR модели"""

    def __init__(self, image_paths: List[Path], labels: List[str], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            image = np.zeros((1000, 1000, 3), dtype=np.uint8)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.labels[idx]

        return image, label, str(image_path)


class PrepareData:
    """Подготовка данных для обучения"""

    def __init__(self, csv_path: Path, images_dir: Path, test_size: float = 0.2, random_state: int = 42):
        self.csv_path = csv_path
        self.images_dir = images_dir
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self._prepare_data()

    def _prepare_data(self):
        # Загружаем данные
        print(f"Загрузка данных из {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        print(f"Загружено {len(self.data)} записей")

        # Создаем словарь для группировки меток по camera_id
        image_to_labels = {}
        missing_count = 0

        for _, row in self.data.iterrows():
            img_name = row["camera_id"]
            label = row["filename"]

            # Проверяем различные возможные расширения
            possible_extensions = ["", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
            img_found = False

            for ext in possible_extensions:
                if ext and img_name.endswith(ext):
                    test_name = img_name
                else:
                    test_name = img_name + ext

                img_path = self.images_dir / test_name
                if img_path.exists():
                    if str(img_path) not in image_to_labels:
                        image_to_labels[str(img_path)] = []
                    image_to_labels[str(img_path)].append(label)
                    img_found = True
                    break

            if not img_found:
                missing_count += 1
                img_path = self.images_dir / img_name
                if img_path.exists():
                    if str(img_path) not in image_to_labels:
                        image_to_labels[str(img_path)] = []
                    image_to_labels[str(img_path)].append(label)
                else:
                    print(f"Изображение не найдено: {img_name}")

        print(
            f"Найдено {len(image_to_labels)} уникальных изображений с {sum(len(labels) for labels in image_to_labels.values())} метками"
        )
        print(f"Пропущено {missing_count} изображений")

        if len(image_to_labels) == 0:
            raise ValueError("Не найдено ни одного валидного изображения!")

        # Создаем списки для разделения
        image_paths = list(image_to_labels.keys())
        labels_lists = list(image_to_labels.values())

        # Для стратификации используем первую метку каждого изображения
        first_labels = [labels[0] for labels in labels_lists]

        # Проверяем, можно ли стратифицировать
        from collections import Counter

        label_counts = Counter(first_labels)
        min_samples = min(label_counts.values()) if label_counts else 0

        if min_samples < 2 or len(set(first_labels)) == len(first_labels):
            stratify = None
            print("Стратификация отключена (уникальные метки или недостаточно образцов)")
        else:
            stratify = first_labels
            print(f"Стратификация включена, классов: {len(set(first_labels))}")

        # Разделяем данные
        (self.train_paths, self.test_paths, self.train_labels, self.test_labels) = train_test_split(
            image_paths, labels_lists, test_size=self.test_size, random_state=self.random_state, stratify=stratify
        )

        print(f"Тренировочный набор: {len(self.train_paths)} изображений")
        print(f"Тестовый набор: {len(self.test_paths)} изображений")

    def get_train_dataset(self):
        # Для обратной совместимости используем первую метку
        train_single_labels = [labels[0] for labels in self.train_labels]
        return OCRDataset([Path(p) for p in self.train_paths], train_single_labels)

    def get_test_dataset(self):
        test_single_labels = [labels[0] for labels in self.test_labels]
        return OCRDataset([Path(p) for p in self.test_paths], test_single_labels)

    def get_train_dataset_with_all_labels(self):
        """Возвращает датасет со всеми метками для каждого изображения"""
        return MultiLabelDataset([Path(p) for p in self.train_paths], self.train_labels)

    def get_test_dataset_with_all_labels(self):
        """Возвращает датасет со всеми метками для каждого изображения"""
        return MultiLabelDataset([Path(p) for p in self.test_paths], self.test_labels)


class MultiLabelDataset(Dataset):
    """Датасет с несколькими метками для каждого изображения"""

    def __init__(self, image_paths: List[Path], labels_lists: List[List[str]], transform=None):
        self.image_paths = image_paths
        self.labels_lists = labels_lists
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            image = np.zeros((1000, 1000, 3), dtype=np.uint8)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        labels = self.labels_lists[idx]

        return image, labels, str(image_path)


class OCRModel:
    """Обертка для OCR системы с возможностью обучения гиперпараметров"""

    def __init__(self):
        self.best_params = None
        self.best_score = float("inf")
        self.study = None

    @staticmethod
    def check_data_quality(dataset, num_samples=5):
        """Проверка качества данных"""
        print("=== ДИАГНОСТИКА ДАННЫХ ===")

        for i in range(min(num_samples, len(dataset))):
            if isinstance(dataset, MultiLabelDataset):
                _, labels, path = dataset[i]
                print(f"{i+1}. {Path(path).name}")
                print(f"   Метки ({len(labels)}): {labels}")
            else:
                _, label, path = dataset[i]
                print(f"{i+1}. {Path(path).name}")
                print(f"   Метка: '{label}'")

            original = cv2.imread(str(path))
            print(f"   Размер оригинала: {original.shape if original is not None else 'N/A'}")

            if original is not None:
                h, w = original.shape[:2]
                roi_height = min(150, h // 5)
                roi = original[h - roi_height : h, : min(600, w)]

                debug_path = f"debug_sample_{i}.jpg"
                cv2.imwrite(debug_path, roi)
                print(f"   Превью сохранено: {debug_path}")
            print("---")

    def find_best_match_distance(self, predicted_text, true_labels):
        """Находит минимальное расстояние Левенштейна между предсказанным текстом и всеми возможными метками"""
        if not true_labels:
            return float("inf"), None

        min_distance = float("inf")
        best_match = None

        for true_label in true_labels:
            distance = levenshtein_distance(str(true_label), predicted_text)
            if distance < min_distance:
                min_distance = distance
                best_match = true_label

        return min_distance, best_match

    def evaluate_params(self, params: Dict[str, Any], dataset, max_samples: int = 200) -> float:
        """Оценка параметров на датасете"""
        try:
            ocr = OverlayOCR(**params)
            total_distance = 0
            count = 0

            # Используем подвыборку для ускорения
            n_samples = min(max_samples, len(dataset))
            indices = np.random.choice(len(dataset), n_samples, replace=False)

            print(f"Оценка параметров на {n_samples} образцах...")

            for i, idx in enumerate(indices):
                if isinstance(dataset, MultiLabelDataset):
                    _, true_labels, image_path = dataset[idx]
                else:
                    _, true_label, image_path = dataset[idx]
                    true_labels = [true_label]  # Преобразуем в список для единообразия

                # Всегда используем оригинальный путь к изображению
                try:
                    final, norm, joined, conf, roi_name = ocr.run_on_image(str(image_path))

                    # Находим наилучшее соответствие среди всех меток
                    distance, best_match = self.find_best_match_distance(final, true_labels)
                    total_distance += distance
                    count += 1

                    if i % 20 == 0:  # Логируем каждые 20 образцов
                        print(
                            f"  [{i+1}/{n_samples}] Лучшее соответствие: '{best_match}' -> '{final}', dist: {distance}"
                        )

                except Exception as e:
                    print(f"  Ошибка при обработке {image_path}: {e}")
                    continue

            avg_distance = total_distance / count if count > 0 else float("inf")
            print(f"  Среднее минимальное расстояние Левенштейна: {avg_distance:.2f}")
            return avg_distance

        except Exception as e:
            print(f"Ошибка оценки параметров: {e}")
            return float("inf")

    def objective(self, trial, train_dataset, max_samples: int):
        """Целевая функция для Optuna"""
        params = {
            "gap_mult": trial.suggest_float("gap_mult", 1.0, 2.0),
            "canvas_size": trial.suggest_categorical("canvas_size", [3600, 4800, 6000]),
            "mag_ratio": trial.suggest_float("mag_ratio", 2.0, 4.0),
            "add_margin": trial.suggest_float("add_margin", 0.05, 0.2),
            "text_threshold": trial.suggest_float("text_threshold", 0.45, 0.7),
            "low_text": trial.suggest_float("low_text", 0.2, 0.4),
            "link_threshold": trial.suggest_float("link_threshold", 0.3, 0.5),
            "langs": ["en"],
            "gpu": torch.cuda.is_available(),
        }

        score = self.evaluate_params(params, train_dataset, max_samples)

        if score < self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            print(f"  Новый лучший результат: {score:.2f}")

        return score

    def train(self, train_dataset, n_trials: int = 50, max_samples: int = 200):
        """Обучение модели"""
        print(f"Запуск оптимизации гиперпараметров ({n_trials} trials)...")

        self.study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

        self.study.optimize(
            lambda trial: self.objective(trial, train_dataset, max_samples), n_trials=n_trials, show_progress_bar=True
        )

        print("Оптимизация завершена!")
        print(f"Лучшие параметры: {self.study.best_params}")
        print(f"Лучшее расстояние: {self.study.best_value:.2f}")

        return self.study.best_params

    def save_model(self, save_path: Path, args=None):
        """Сохранение модели и параметров"""
        if self.best_params is None:
            raise ValueError("Модель не обучена. Сначала вызовите train()")

        save_path.mkdir(parents=True, exist_ok=True)

        if args is not None:
            params_filename = args.ocr_model_params
            study_filename = args.optuna_study
        else:
            params_filename = "ocr_model_params.json"
            study_filename = "optuna_study.pkl"

        model_info = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "study_trials": len(self.study.trials) if self.study else 0,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        with open(save_path / params_filename, "w") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        if self.study:
            joblib.dump(self.study, save_path / study_filename)

        print(f"Модель сохранена в: {save_path}")

    def load_model(self, load_path: Path, args=None):
        """Загрузка модели"""
        if args is not None:
            params_filename = args.ocr_model_params
            study_filename = args.optuna_study
        else:
            params_filename = "ocr_model_params.json"
            study_filename = "optuna_study.pkl"

        with open(load_path / params_filename, "r") as f:
            model_info = json.load(f)

        self.best_params = model_info["best_params"]
        self.best_score = model_info["best_score"]

        study_path = load_path / study_filename
        if study_path.exists():
            self.study = joblib.load(study_path)

        return OverlayOCR(**self.best_params)


def find_file_by_pattern(directory, pattern):
    """
    Ищет файл в директории по шаблону имени.
    Возвращает Path к первому найденному файлу или None.
    """
    path = Path(directory)
    if not path.exists():
        return None
    for file_path in path.iterdir():
        if file_path.is_file() and pattern in file_path.name:
            return file_path
    return None


def find_dir_by_pattern(directory, pattern):
    """
    Ищет директорию по шаблону имени.
    Возвращает Path к первой найденной директории или None.
    """
    path = Path(directory)
    if not path.exists():
        return None
    for dir_path in path.iterdir():
        if dir_path.is_dir() and pattern in dir_path.name:
            return dir_path
    return None


def main():
    """Основная функция обучения"""
    args = parse_args()

    try:
        # Диагностика путей в DataSphere
        print("\n" + "=" * 60)
        print("DATASPHERE PATH DIAGNOSTICS:")
        print(f"Original CSV path: {args.csv_path}")
        print(f"Original Images dir: {args.images_dir}")

        # 1. Определяем корневую директорию для поиска
        search_root = ROOT_DIR

        # 2. Гибкий поиск CSV-файла
        csv_path = Path(args.csv_path)
        if not csv_path.exists():
            # Пробуем найти файл, содержащий в имени ключевые слова
            possible_csv = find_file_by_pattern(search_root, "csv_file")
            if possible_csv:
                csv_path = possible_csv
                print(f"Найден CSV-файл по шаблону: {csv_path}")
            else:
                # Если по шаблону не нашли, пробуем просто взять первый файл в корне с расширением .csv
                for item in search_root.iterdir():
                    if item.is_file() and item.suffix.lower() == ".csv":
                        csv_path = item
                        print(f"Найден CSV-файл по расширению: {csv_path}")
                        break
                else:
                    raise FileNotFoundError(
                        f"CSV файл не найден: {args.csv_path}. Доступные файлы в {search_root}: {list(search_root.iterdir())}"
                    )

        # 3. Гибкий поиск директории с изображениями
        images_dir = Path(args.images_dir)
        if not images_dir.exists():
            # Пробуем найти директорию, содержащую в имени ключевые слова
            possible_images_dir = find_dir_by_pattern(search_root, "images_dir")
            if possible_images_dir:
                images_dir = possible_images_dir
                print(f"Найдена директория с изображениями по шаблону: {images_dir}")
            else:
                raise FileNotFoundError(
                    f"Директория с изображениями не найдена: {args.images_dir}. Доступные директории в {search_root}: {[d.name for d in search_root.iterdir() if d.is_dir()]}"
                )

        print(f"Final CSV path: {csv_path}")
        print(f"Final Images dir: {images_dir}")
        print(f"CSV exists: {csv_path.exists()}")
        print(f"Images dir exists: {images_dir.exists()}")
        print("=" * 60 + "\n")

        # Создание датасета с исправленными путями
        print("Подготовка данных...")
        dataset = PrepareData(csv_path=csv_path, images_dir=images_dir, test_size=0.2, random_state=42)

        # Получаем датасеты со всеми метками
        train_dataset = dataset.get_train_dataset_with_all_labels()
        test_dataset = dataset.get_test_dataset_with_all_labels()

        print(f"Размер тренировочного датасета: {len(train_dataset)}")
        print(f"Размер тестового датасета: {len(test_dataset)}")

        # Диагностика данных перед обучением
        OCRModel.check_data_quality(train_dataset, num_samples=3)

        # Тестовый прогон на одном изображении
        if len(train_dataset) > 0:
            _, test_labels, test_path = train_dataset[0]
            print(f"\nТестовый прогон на первом изображении:")
            print(f"Путь: {test_path}")
            print(f"Возможные метки ({len(test_labels)}): {test_labels}")

            # Проверяем базовый OCR
            ocr = OverlayOCR()
            try:
                final, norm, joined, conf, roi_name = ocr.run_on_image(str(test_path))
                distance, best_match = OCRModel().find_best_match_distance(final, test_labels)
                print(f"Результат OCR: '{final}'")
                print(f"Лучшее соответствие: '{best_match}', расстояние: {distance}")

                if distance > 10:
                    print("ВНИМАНИЕ: Большая ошибка на тестовом изображении!")
            except Exception as e:
                print(f"Ошибка при тестовом OCR: {e}")

        # Создаем и обучаем модель
        model = OCRModel()

        # Обучаем на тренировочных данных со всеми метками
        best_params = model.train(train_dataset, n_trials=args.n_trials, max_samples=args.max_samples)

        # Сохраняем модель в текущую директорию (для DataSphere)
        model.save_model(ROOT_DIR, args)

        # Тестируем на тестовых данных со всеми метками
        print("\nТестирование на тестовом наборе...")
        test_ocr = OverlayOCR(**best_params)
        test_distances = []
        best_matches = []
        test_samples = min(50, len(test_dataset))

        for i in range(test_samples):
            _, true_labels, image_path = test_dataset[i]

            try:
                final, norm, joined, conf, roi_name = test_ocr.run_on_image(str(image_path))
                distance, best_match = model.find_best_match_distance(final, true_labels)
                test_distances.append(distance)
                best_matches.append(best_match)

                if i % 10 == 0:
                    print(f"Тест [{i+1}/{test_samples}]: '{best_match}' -> '{final}', dist: {distance}")
            except Exception as e:
                print(f"Ошибка при тестировании {image_path}: {e}")
                continue

        if test_distances:
            avg_test_distance = np.mean(test_distances)
            std_test_distance = np.std(test_distances)

            # Анализ результатов
            perfect_matches = sum(1 for d in test_distances if d == 0)
            good_matches = sum(1 for d in test_distances if d <= 2)

            print(f"\nРезультаты тестирования:")
            print(f"Среднее минимальное расстояние: {avg_test_distance:.2f}")
            print(f"Стандартное отклонение: {std_test_distance:.2f}")
            print(f"Идеальные совпадения (расстояние=0): {perfect_matches}/{len(test_distances)}")
            print(f"Хорошие совпадения (расстояние≤2): {good_matches}/{len(test_distances)}")
            print(f"Минимальное расстояние: {np.min(test_distances):.2f}")
            print(f"Максимальное расстояние: {np.max(test_distances):.2f}")

            # Сохраняем результаты тестирования
            results = {
                "test_avg_distance": avg_test_distance,
                "test_std_distance": std_test_distance,
                "perfect_matches": perfect_matches,
                "good_matches": good_matches,
                "test_samples_evaluated": len(test_distances),
                "train_dataset_size": len(train_dataset),
                "test_dataset_size": len(test_dataset),
                "best_params": best_params,
            }

            with open(ROOT_DIR / "test_results.json", "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            print("Не удалось получить результаты тестирования!")

        print(f"\nОбучение завершено! Результаты сохранены в: {ROOT_DIR.absolute()}")

    except Exception as e:
        print(f"Критическая ошибка в main: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
