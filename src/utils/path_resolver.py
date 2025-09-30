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
import sys
from pathlib import Path


def setup_project_paths():
    """
    Настраивает пути проекта для корректной работы импортов
    в различных средах (локально, Docker, облаке)
    """
    current_file = Path(__file__).resolve()

    # Определяем корень проекта
    # Пытаемся найти корень по маркерам (например, наличие setup.py или pyproject.toml)
    project_root = current_file.parent.parent
    markers = ["setup.py", "pyproject.toml", "requirements.txt", ".git"]

    # Поиск корня проекта
    temp_root = current_file
    while temp_root != temp_root.parent:
        if any((temp_root / marker).exists() for marker in markers):
            project_root = temp_root
            break
        temp_root = temp_root.parent

    # Альтернативный способ: явное указание через переменную окружения
    env_root = Path.cwd()  # Можно переопределить через переменную окружения если нужно

    # Выбираем финальный корень
    final_root = project_root

    # Добавляем необходимые пути в sys.path
    paths_to_add = [
        final_root,  # Корень проекта
        final_root / "src",  # Папка src
        final_root / "src" / "utils",  # Утилиты
        final_root / "src" / "geo",  # Гео модули
        final_root / "src" / "models",  # Модели
        final_root / "src" / "data",  # Данные
        final_root / "src" / "engine",  # Движок
    ]

    for path in paths_to_add:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    return final_root


# Автоматический вызов при импорте
PROJECT_ROOT = setup_project_paths()
