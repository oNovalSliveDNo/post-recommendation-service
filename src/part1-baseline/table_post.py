from sqlalchemy import Column, Integer, String, Text

from database import Base, SessionLocal


class Post(Base):
    __tablename__ = "post"  # Имя таблицы в базе данных

    id = Column(Integer, primary_key=True)  # Первичный ключ
    text = Column(Text)  # Содержимое поста
    topic = Column(String)  # Заголовок поста


if __name__ == "__main__":
    # Создание сессии
    session = SessionLocal()

    try:
        # Запрос для получения первых 10 id постов с темой "business"
        posts = (
            session.query(Post.id)
            .filter(Post.topic == "business")
            .order_by(Post.id.desc())
            .limit(10)
            .all()
        )

        # Извлечение id из результата
        post_ids = [post[0] for post in posts]  # Используем list comprehension

        # Печать результата
        print(post_ids)

    finally:
        session.close()
