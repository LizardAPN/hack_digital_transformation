-- =============================================
-- Скрипт: Инициализация таблиц базы данных
-- Автор: Система автоматического определения координат зданий
-- Дата: 2025-09-30
-- =============================================

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', 'public', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;
SET search_path TO public;

-- Создание таблицы пользователей
CREATE TABLE IF NOT EXISTS public.users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    session_token VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Создание таблицы загруженных фотографий
CREATE TABLE IF NOT EXISTS public.user_photos (
    id SERIAL PRIMARY KEY,
    owner_id INTEGER NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    photo_url VARCHAR(512) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Создание таблицы результатов обработки изображений
CREATE TABLE IF NOT EXISTS public.processing_results (
    id SERIAL PRIMARY KEY,
    image_path VARCHAR(512) NOT NULL,
    task_id VARCHAR(255),
    request_id VARCHAR(255),
    coordinates JSONB,
    address TEXT,
    ocr_result JSONB,
    buildings JSONB,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Создание индексов для оптимизации запросов
CREATE INDEX IF NOT EXISTS idx_user_photos_owner_id ON public.user_photos(owner_id);
CREATE INDEX IF NOT EXISTS idx_processing_results_task_id ON public.processing_results(task_id);
CREATE INDEX IF NOT EXISTS idx_processing_results_request_id ON public.processing_results(request_id);
CREATE INDEX IF NOT EXISTS idx_processing_results_processed_at ON public.processing_results(processed_at);

-- Функция для автоматического обновления поля updated_at
CREATE OR REPLACE FUNCTION public.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Триггеры для автоматического обновления поля updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON public.users FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_user_photos_updated_at BEFORE UPDATE ON public.user_photos FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
CREATE TRIGGER update_processing_results_updated_at BEFORE UPDATE ON public.processing_results FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();