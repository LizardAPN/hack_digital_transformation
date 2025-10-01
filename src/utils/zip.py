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
from typing import List, Optional
import os
import zipfile


def extract_zip_advanced(zip_path: str, extract_to: str, remove_after_extract: bool = False) -> Optional[List[str]]:
    """
    Разархивирует ZIP файл с обработкой ошибок и опцией удаления архива.

    Функция извлекает содержимое ZIP архива в указанную директорию с полной 
    обработкой ошибок. Поддерживает проверку целостности архива и опциональное 
    удаление исходного архива после успешного извлечения.

    Параметры
    ----------
    zip_path : str
        Путь к ZIP файлу, который нужно разархивировать.
    extract_to : str
        Директория, в которую будет извлечено содержимое архива.
    remove_after_extract : bool, optional
        Флаг, указывающий на необходимость удаления архива после 
        успешного извлечения (по умолчанию False).

    Возвращает
    -------
    list или None
        Список имен файлов, извлеченных из архива, или None в случае ошибки.

    Исключения
    ----------
    FileNotFoundError
        Возникает, если указанный ZIP файл не найден.
    ValueError
        Возникает, если файл не является корректным ZIP архивом.
    zipfile.BadZipFile
        Возникает, если архив поврежден или не является ZIP файлом.

    Примеры
    --------
    >>> file_list = extract_zip_advanced("archive.zip", "extracted_files")
    >>> if file_list is not None:
    ...     print(f"Извлечено {len(file_list)} файлов")
    >>> 
    >>> # С удалением архива после извлечения
    >>> file_list = extract_zip_advanced(
    ...     "archive.zip", 
    ...     "extracted_files", 
    ...     remove_after_extract=True
    ... )
    """
    try:
        # Проверяем существование архива
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP файл не найден: {zip_path}")

        # Проверяем, что это действительно ZIP файл
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"Файл не является ZIP архивом: {zip_path}")

        # Создаем директорию для извлечения
        os.makedirs(extract_to, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Получаем информацию о файлах в архиве
            file_list: List[str] = zip_ref.namelist()
            print(f"Файлы в архиве: {file_list}")

            # Извлекаем все файлы
            zip_ref.extractall(extract_to)

            print(f"Архив успешно разархивирован в {extract_to}")
            print(f"Извлечено {len(file_list)} файлов")

        # Удаляем архив после успешного извлечения с дополнительной проверкой
        if remove_after_extract:
            # Проверяем, что файлы действительно извлечены
            all_files_extracted = True
            for file in file_list:
                extracted_file_path = os.path.join(extract_to, file)
                if not os.path.exists(extracted_file_path):
                    print(f"Предупреждение: файл {file} не был извлечен")
                    all_files_extracted = False

            if all_files_extracted:
                os.remove(zip_path)
                print(f"Архив {zip_path} удален после успешного извлечения")
            else:
                print(f"Архив {zip_path} не удален: не все файлы извлечены корректно")

        return file_list

    except zipfile.BadZipFile:
        print(f"Ошибка: архив поврежден или не является ZIP файлом")
        return None
    except Exception as e:
        print(f"Ошибка при разархивировании: {e}")
        return None
