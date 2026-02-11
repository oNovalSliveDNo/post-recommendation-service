import os
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv


@contextmanager
def get_db_connection(conn_str):
    """Контекстный менеджер для подключения к БД"""
    conn = psycopg2.connect(conn_str)
    try:
        yield conn
    finally:
        conn.close()


def copy_query_to_df(query, conn_str):
    """Выполняет COPY запрос и возвращает DataFrame"""
    with get_db_connection(conn_str) as conn:
        with conn.cursor() as cur:
            buffer = StringIO()
            # Используем FORCE QUOTE * чтобы обработать все поля
            cur.copy_expert(
                f"COPY ({query}) TO STDOUT WITH (FORMAT CSV, HEADER, DELIMITER ',')",
                buffer
            )
            buffer.seek(0)
            df = pd.read_csv(buffer)
    return df


def save_dataframe_to_csv(df, filepath, index=False):
    """Сохраняет DataFrame в CSV файл"""
    # Создаем директорию если не существует
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    print(f"Сохранено: {filepath} ({len(df)} строк)")


def main():
    """Основная функция для выгрузки данных"""
    # Загружаем переменные окружения
    load_dotenv()
    DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL")

    if not DATABASE_URL:
        raise ValueError("SQLALCHEMY_DATABASE_URL не найден в переменных окружения")

    # Определяем пути
    PROJECT_ROOT = Path().resolve().parent
    DATA_DIR = PROJECT_ROOT / "demo_data"

    # Создаем словарь запросов с именами таблиц как в базе
    queries = {
        "user_1000": "SELECT * FROM public.user LIMIT 1000",
        "post_1000": "SELECT * FROM public.post LIMIT 1000",
        "feed_action_1000": "SELECT * FROM public.feed_action LIMIT 1000"
    }

    # Словарь для хранения результатов
    datasets = {}

    # Выгружаем данные и замеряем время
    print("Начинаем выгрузку данных из PostgreSQL...")
    print("-" * 50)

    for name, query in queries.items():
        print(f"Выгружаем таблицу: {name}")
        print(f"Запрос: {query[:80]}...")

        start_time = time.time()

        try:
            # Загрузка данных из БД
            df = copy_query_to_df(query, DATABASE_URL)
            load_time = time.time() - start_time

            # Сохранение в CSV
            save_start_time = time.time()
            filename = DATA_DIR / f"{name}.csv"
            save_dataframe_to_csv(df, filename, index=False)
            save_time = time.time() - save_start_time

            # Общее время для этой таблицы
            total_time = time.time() - start_time

            datasets[name] = df
            print(f"✓ Выгружено: {len(df)} строк")
            print(f"  Время загрузки из БД: {load_time:.2f} секунд")
            print(f"  Время сохранения в CSV: {save_time:.2f} секунд")
            print(f"  Общее время для таблицы: {total_time:.2f} секунд")
            print()

        except Exception as e:
            print(f"✗ Ошибка при выгрузке {name}: {e}")
            print()

    # Выводим статистику
    print("\n" + "=" * 50)
    print("ВЫГРУЗКА ЗАВЕРШЕНА")
    print("=" * 50)
    for name, df in datasets.items():
        print(f"{name}: {len(df)} строк, {len(df.columns)} колонок")


if __name__ == "__main__":
    main()
