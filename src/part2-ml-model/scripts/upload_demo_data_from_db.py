import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


def create_db_engine(SQLALCHEMY_DATABASE_URL):
    """
    Создаёт и возвращает объект подключения к базе данных PostgreSQL.
    """
    return create_engine(SQLALCHEMY_DATABASE_URL)


def fetch_data(query, engine):
    """
    Выполняет SQL-запрос и возвращает результат в виде DataFrame.

    :param query: SQL-запрос в виде строки
    :param engine: объект подключения к базе данных
    :return: DataFrame с результатами запроса
    """
    return pd.read_sql(query, con=engine)


def save_to_csv(df, filename, index=False):
    """
    Сохраняет DataFrame в CSV-файл.

    :param df: DataFrame, который нужно сохранить
    :param filename: имя файла для сохранения
    :param index: флаг, указывающий на сохранение индексов (True/False)
    """
    df.to_csv(f'{filename}', index=index)


def main():
    """
    Основная функция скрипта:
    1. Устанавливает соединение с базой данных.
    2. Выполняет SQL-запросы для получения данных.
    3. Сохраняет полученные данные в CSV-файлы с и без индексов.
    """
    load_dotenv()
    SQLALCHEMY_DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL")

    # Корень репозитория
    PROJECT_ROOT = Path().resolve().parent

    # Папка с данными
    DATA_DIR = PROJECT_ROOT / "demo data"

    engine = create_db_engine(SQLALCHEMY_DATABASE_URL)

    # Словарь с SQL-запросами
    queries = {
        "user_data": "SELECT * FROM public.user_data LIMIT 1000",
        "post_text_df": "SELECT * FROM public.post_text_df LIMIT 1000",
        "feed_data": "SELECT * FROM public.feed_data LIMIT 1000"
    }

    # Выгружаем данные из базы
    datasets = {name: fetch_data(query, engine) for name, query in queries.items()}

    # Сохраняем данные в файлы с индексом и без
    for name, df in datasets.items():
        save_to_csv(df, DATA_DIR / f"{name}.csv", index=False)

    print("Данные успешно выгружены в CSV файлы!")


if __name__ == "__main__":
    main()
