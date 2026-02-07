from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# Модель для представления данных из таблицы user
class UserGet(BaseModel):
    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source: str

    class Config:
        from_attributes = True


# Модель для представления данных из таблицы post
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        from_attributes = True


# Модель для представления данных из таблицы feed_action
class FeedGet(BaseModel):
    user_id: int
    post_id: int
    action: str
    time: datetime

    user: Optional[UserGet]  # Добавляем вложенную модель UserGet
    post: Optional[PostGet]  # Добавляем вложенную модель PostGet

    class Config:
        from_attributes = True
