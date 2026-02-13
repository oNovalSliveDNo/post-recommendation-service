import os
from datetime import datetime
from typing import List

import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine

load_dotenv()  # Загружает переменные окружения
SQLALCHEMY_DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL")  # Строка подключения к PostgreSQL


# Модель данных ответа API
class PostGet(BaseModel):
    """
    Описывает структуру одного рекомендованного поста.
    """
    id: int
    text: str
    topic: str

    class Config:
        """
        Разрешает создавать модель из ORM-объектов, из namedtuple, из генераторов.
        """
        from_attributes = True


# Функция выбора пути к модели
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = "/workdir/user_input/model"
    else:
        MODEL_PATH = path
    return MODEL_PATH


# Функция загрузки модели
def load_models():
    model_path = get_model_path("catboost_dl_model.cbm")
    model = CatBoostClassifier()  # Создаёт пустую модель
    model.load_model(model_path)  # Загружает веса
    return model  # Возвращает объект модели


# Функция загрузки данных из БД с обработкой больших объёмов
def batch_load_sql(query: str) -> pd.DataFrame:
    """
    Универсальная функция безопасной загрузки больших таблиц.
    """
    CHUNKSIZE = 200000  # Подобран под лимит RAM 4 Гб
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    conn = engine.connect().execution_options(stream_results=True)  # данные не буферизуются целиком

    chunks = []
    # Таблица читается порциями
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        # Каждая порция — маленький DataFrame
        chunks.append(chunk_dataframe)

    conn.close()

    return pd.concat(chunks, ignore_index=True)


# Функция загрузки всех необходимых данных
def load_features():
    """
    Единая точка загрузки всех данных, которые понадобятся сервису.
    """
    tables = {
        "user_data": "processed_user_data",  # user-фичи
        "feed_data": "processed_feed_data_like",  # история лайков
        "post_features": "processed_post_text_dl",  # post-фичи для ML
        "clear_post_text": "clear_post_text"  # текст для ответа API
    }

    # Делаем долгую инициализацию, потому что
    # Запросы потом должны быть быстрыми
    user_data = batch_load_sql(f'SELECT * FROM "{tables["user_data"]}"')
    feed_data = batch_load_sql(f'SELECT * FROM "{tables["feed_data"]}"')
    post_features = batch_load_sql(f'SELECT * FROM "{tables["post_features"]}"')
    clear_post_text = batch_load_sql(f'SELECT * FROM "{tables["clear_post_text"]}"')

    return user_data, feed_data, post_features, clear_post_text


# Инициализация FastAPI
app = FastAPI()
model = load_models()  # загружается модель
user_data, feed_data, post_features, clear_post_text = load_features()  # загружаются все фичи


# --- Дальше сервис только читает память (in-memory ranking engine) ---

# Функция предобработки данных перед подачей в модель
def preprocessing_data(user_id, month, day):
    # Достаём данные пользователя (один user_id → одна строка)
    single_user_features = user_data.loc[user_data['user_id'] == user_id]

    # --- Фильтруем посты, исключая уже лайкнутые пользователем ---

    # Оставляем только посты, которые кто-то когда-то лайкнул
    filtered_post_text = post_features[post_features['post_id'].isin(feed_data['post_id'].unique())]
    # Убираем лайкнутые пользователем посты
    # Берём историю лайков пользователя и исключаем эти посты из кандидатов
    unique_post_ids = feed_data[feed_data['user_id'] == user_id]['post_id'].unique().tolist()
    filtered_post_text_for_user = filtered_post_text[~filtered_post_text['post_id'].isin(unique_post_ids)]

    # Добавляем информацию о пользователе
    for col in user_data.columns:
        # Добавляем user-фичи (возраст, город, пол, ...) к каждому посту
        filtered_post_text_for_user[col] = single_user_features[col].iloc[0]

    # Добавляем нормализованные данные о времени
    filtered_post_text_for_user['month'] = month / 12  # нормализация к [0; 1]
    filtered_post_text_for_user['day'] = day / 31  # нормализация к [0; 1]

    # Оставляем только нужные признаки
    feature_columns = ['user_id', 'post_id',
                       'month', 'day',
                       'gender', 'os', 'source', 'city_countscaled', 'country_countscaled',
                       'age_group_18_20', 'age_group_20_22', 'age_group_22_25', 'age_group_25_30', 'age_group_30_35',
                       'age_group_35_40', 'age_group_gt_40',
                       'exp_group_1', 'exp_group_2', 'exp_group_3', 'exp_group_4',
                       'text_cluster',
                       'distance_to_cluster_0', 'distance_to_cluster_1', 'distance_to_cluster_2',
                       'distance_to_cluster_3',
                       'distance_to_cluster_4', 'distance_to_cluster_5', 'distance_to_cluster_6',
                       'distance_to_cluster_7',
                       'distance_to_cluster_8', 'distance_to_cluster_9', 'distance_to_cluster_10',
                       'distance_to_cluster_11',
                       'distance_to_cluster_12', 'distance_to_cluster_13', 'distance_to_cluster_14',
                       'topic_covid', 'topic_entertainment', 'topic_movie', 'topic_politics', 'topic_sport',
                       'topic_tech']

    # Отбор ровно тех фичей, что были на обучении
    return filtered_post_text_for_user[feature_columns]


# Основной эндпоинт рекомендаций
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    # Подготавливаем данные
    # Из (user_id + time) получается DataFrame (candidate_posts × features)
    data = preprocessing_data(id, time.month, time.day)

    # Делаем предсказание вероятности лайка
    # Модель не выбирает посты, а ранжирует кандидатов
    data['pred_proba'] = model.predict_proba(data)[:, 1]

    # Отфильтруем нужные строки
    # Сортировка по вероятности лайка и выбор top-K
    post_ids = list(data.sort_values('pred_proba', ascending=False).head(limit)['post_id'])

    # Используем генератор, чтобы не загружать весь DataFrame в память
    # Не делаем merge и не создаём новый DataFrame -> экономим память.
    def stream_post_data(df, post_ids):
        for row in df.itertuples(index=False):
            if row.post_id in post_ids:
                yield {"id": row.post_id, "text": row.text, "topic": row.topic}

    # Генерируем JSON-ответ построчно
    # Берём текст и topic, формируем JSON, возвращаем клиенту
    return list(stream_post_data(clear_post_text, set(post_ids)))

# Архитектура сервиса:
# 1. Предварительный отбор кандидатов (candidate filtering)
# 2. ML-ранжирование кандидатов
# 3. Быстрый in-memory inference без SQL внутри эндпоинта
