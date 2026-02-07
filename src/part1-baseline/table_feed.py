from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from database import Base
from table_post import Post
from table_user import User


class Feed(Base):
    __tablename__ = "feed_action"

    user_id = Column(Integer, ForeignKey(User.id), primary_key=True)  # Внешний ключ на User
    post_id = Column(Integer, ForeignKey(Post.id), primary_key=True)  # Внешний ключ на Post
    action = Column(String)
    time = Column(DateTime)

    # Определяем отношения
    user = relationship("User")  # Связь с таблицей User
    post = relationship("Post")  # Связь с таблицей Post
