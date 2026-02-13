import hashlib
import os
from datetime import datetime
from typing import List

import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger
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


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


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
        logger.info(f"Got chunk: {len(chunk_dataframe)}")

    conn.close()

    return pd.concat(chunks, ignore_index=True)


def load_common_features():
    logger.info('SELECT * FROM "processed_user_data"')
    user_data = batch_load_sql(f'SELECT * FROM "processed_user_data"')

    logger.info('SELECT * FROM "processed_feed_data_like"')
    feed_data = batch_load_sql(f'SELECT * FROM "processed_feed_data_like"')

    logger.info('SELECT * FROM "clear_post_text"')
    clear_post_text = batch_load_sql(f'SELECT * FROM "clear_post_text"')

    logger.info('SELECT * FROM "processed_post_text_ml"')
    control_post_text = batch_load_sql(f'SELECT * FROM "processed_post_text_ml"')

    logger.info('SELECT * FROM "processed_post_text_dl"')
    test_post_text = batch_load_sql(f'SELECT * FROM "processed_post_text_dl"')

    return user_data, feed_data, clear_post_text, control_post_text, test_post_text


def get_model_path(model_version: str) -> str:
    # print(os.environ)
    if (os.environ.get("IS_LMS") == "1"):
        model_path = f"/workdir/user_input/model_{model_version}"
    else:
        model_path = f"model_{model_version}.cbm"

    return model_path


def load_models(model_version: str):
    model_path = get_model_path(model_version)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    print(f"Модель загружена: {model_version} (путь: {model_path})")  # Вывод информации о модели
    return loaded_model


app = FastAPI()
user_data, feed_data, clear_post_text, control_post_text, test_post_text = load_common_features()

# Теперь мы загружаем сразу 2 модели
model_control = load_models("control")
model_test = load_models("test")

SALT = "my_salt"


def get_user_group(id: int) -> str:
    value_str = str(id) + SALT
    value_num = int(hashlib.md5(value_str.encode()).hexdigest(), 16)
    percent = value_num % 100
    if percent < 50:
        return "control"
    elif percent < 100:
        return "test"
    return "unknown"


def calculate_features(user_id: int, time: datetime, group: str) -> pd.DataFrame:
    # Достаём данные пользователя
    single_user_features = user_data.loc[user_data['user_id'] == user_id]

    post_text = None

    if group == 'control':
        post_text = control_post_text

    elif group == 'test':
        post_text = test_post_text

    else:
        raise ValueError("unknown group")

    # Фильтруем посты, исключая уже просмотренные
    filtered_post_text = post_text[post_text['post_id'].isin(feed_data['post_id'].unique())]
    unique_post_ids = feed_data[feed_data['user_id'] == user_id]['post_id'].unique().tolist()
    filtered_post_text_for_user = filtered_post_text[~filtered_post_text['post_id'].isin(unique_post_ids)]

    # Добавляем информацию о пользователе
    for col in user_data.columns:
        filtered_post_text_for_user[col] = single_user_features[col].iloc[0]

    # Добавляем нормализованные данные о времени
    filtered_post_text_for_user['month'] = time.month / 12
    filtered_post_text_for_user['day'] = time.day / 31

    feature_columns = list()

    # Оставляем только нужные признаки
    if group == 'control':
        feature_columns = ['user_id', 'post_id', 'month', 'day', 'gender', 'os', 'source', 'city_countscaled',
                           'country_countscaled', 'age_group_18_20', 'age_group_20_22', 'age_group_22_25',
                           'age_group_25_30', 'age_group_30_35', 'age_group_35_40', 'age_group_gt_40', 'exp_group_1',
                           'exp_group_2', 'exp_group_3', 'exp_group_4', 'tfidf_pc1', 'tfidf_pc2', 'tfidf_pc3',
                           'tfidf_pc4', 'tfidf_pc5', 'tfidf_pc6', 'tfidf_pc7', 'tfidf_pc8', 'tfidf_pc9', 'tfidf_pc10',
                           'tfidf_pc11', 'tfidf_pc12', 'tfidf_pc13', 'tfidf_pc14', 'tfidf_pc15', 'tfidf_pc16',
                           'tfidf_pc17',
                           'tfidf_pc18', 'tfidf_pc19', 'tfidf_pc20', 'topic_covid', 'topic_entertainment',
                           'topic_movie',
                           'topic_politics', 'topic_sport', 'topic_tech']

    elif group == 'test':
        feature_columns = ['user_id', 'post_id',
                           'month', 'day',
                           'gender', 'os', 'source', 'city_countscaled', 'country_countscaled',
                           'age_group_18_20', 'age_group_20_22', 'age_group_22_25', 'age_group_25_30',
                           'age_group_30_35',
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

    else:
        raise ValueError("unknown group")

    return filtered_post_text_for_user[feature_columns]


def get_recommended_feed(id: int, time: datetime, limit: int) -> Response:
    # Выбираем группу пользователи
    user_group = get_user_group(id=id)
    logger.info(f"user group {user_group}")

    # Выбираем нужную модель
    if user_group == "control":
        model = model_control
    elif user_group == "test":
        model = model_test
    else:
        raise ValueError("unknown group")

    # Подготавливаем данные
    data = calculate_features(user_id=id, time=time, group=user_group)

    # Делаем предсказание вероятности лайка
    logger.info("predicting")
    data['pred_proba'] = model.predict_proba(data)[:, 1]

    # Отфильтруем нужные строки
    post_ids = list(data.sort_values('pred_proba', ascending=False).head(limit)['post_id'])

    # Используем генератор, чтобы не загружать весь DataFrame в память
    def stream_post_data(df, post_ids):
        for row in df.itertuples(index=False):
            if row.post_id in post_ids:
                yield {"id": row.post_id, "text": row.text, "topic": row.topic}

    # Генерируем JSON-ответ построчно
    return Response(
        recommendations=list(stream_post_data(clear_post_text, set(post_ids))),
        exp_group=user_group,
    )


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    return get_recommended_feed(id, time, limit)
