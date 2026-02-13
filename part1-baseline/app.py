from typing import List

from fastapi import FastAPI, HTTPException, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import SessionLocal
from schema import UserGet, PostGet, FeedGet
from table_feed import Feed  # Импортируем модель Feed из соответствующего модуля
from table_post import Post
from table_user import User

# Создаем экземпляр приложения FastAPI
app = FastAPI()


# Зависимость, которая создает сессию базы данных для каждого запроса и закрывает её после завершения
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Эндпоинт для получения данных пользователя по ID
@app.get("/user/{id}", response_model=UserGet)
def get_user(id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


# Эндпоинт для получения данных поста по ID
@app.get("/post/{id}", response_model=PostGet)
def get_post(id: int, db: Session = Depends(get_db)):
    post = db.query(Post).filter(Post.id == id).first()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    return post


# Эндпоинт для получения фида действий пользователя по ID
@app.get("/user/{id}/feed", response_model=List[FeedGet])
def get_user_feed(id: int, limit: int = Query(10), db: Session = Depends(get_db)):
    feed = (
        db.query(Feed)
        .filter(Feed.user_id == id)
        .order_by(Feed.time.desc())
        .limit(limit)
        .all()
    )
    return feed


# Эндпоинт для получения фида действий по посту по ID
@app.get("/post/{id}/feed", response_model=List[FeedGet])
def get_post_feed(id: int, limit: int = Query(10), db: Session = Depends(get_db)):
    feed = (
        db.query(Feed)
        .filter(Feed.post_id == id)
        .order_by(Feed.time.desc())
        .limit(limit)
        .all()
    )
    return feed


@app.get("/post/recommendations/", response_model=List[PostGet])
def get_recommended_posts(limit: int = Query(10), db: Session = Depends(get_db)):
    recommended_posts = (
        db.query(Post)
        .select_from(Feed)
        .join(Post, Feed.post_id == Post.id)
        .filter(Feed.action == 'like')
        .group_by(Post.id)
        .order_by(func.count(Feed.post_id).desc())
        .limit(limit)
        .all()
    )
    return recommended_posts
