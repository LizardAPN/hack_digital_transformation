#!/usr/bin/env python3
"""
Скрипт для добавления универсального импорта во все Python файлы в папке src
"""

import os
import sys
from pathlib import Path

# Шаблон универсального импорта
UNIVERSAL_IMPORT_TEMPLATE = '''import sys
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

'''

def needs_universal_import(file_path):
    """Проверяет, нуждается ли файл в универсальном импорте"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем, содержит ли файл уже универсальный импорт
    if "from path_resolver import setup_project_paths" in content:
        return False
    
    # Проверяем, содержит ли файл относительные импорты
    if "from .." in content:
        return True
    
    # Проверяем, содержит ли файл абсолютные импорты из src
    if "from src." in content:
        return True
    
    # Для остальных файлов добавляем универсальный импорт
    return True

def add_universal_import(file_path):
    """Добавляет универсальный импорт в файл"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем, есть ли уже универсальный импорт
    if "from path_resolver import setup_project_paths" in content:
        print(f"  Файл {file_path} уже содержит универсальный импорт")
        return False
    
    # Разделяем содержимое на строки
    lines = content.split('\n')
    
    # Находим первую строку, которая не является комментарием или пустой
    insert_index = 0
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped and not line_stripped.startswith('#'):
            insert_index = i
            break
    
    # Вставляем универсальный импорт перед первой строкой кода
    lines.insert(insert_index, UNIVERSAL_IMPORT_TEMPLATE.rstrip())
    
    # Записываем обновленное содержимое обратно в файл
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"  Добавлен универсальный импорт в {file_path}")
    return True

def process_src_directory(src_path):
    """Обрабатывает все Python файлы в директории src"""
    print(f"Обработка файлов в директории: {src_path}")
    
    # Получаем список всех Python файлов
    python_files = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Найдено Python файлов: {len(python_files)}")
    
    # Сортируем файлы для консистентности
    python_files.sort()
    
    # Обрабатываем каждый файл
    modified_count = 0
    for file_path in python_files:
        print(f"Обработка {file_path}...")
        if add_universal_import(file_path):
            modified_count += 1
    
    print(f"\nОбработано файлов: {len(python_files)}")
    print(f"Изменено файлов: {modified_count}")

def main():
    """Основная функция"""
    # Определяем путь к директории src
    project_root = Path(__file__).resolve().parent
    src_path = project_root / "src"
    
    if not src_path.exists():
        print(f"Директория {src_path} не найдена")
        return 1
    
    print("Добавление универсальных импортов во все Python файлы в папке src")
    print("=" * 60)
    
    process_src_directory(src_path)
    
    print("=" * 60)
    print("Завершено!")

if __name__ == "__main__":
    main()
