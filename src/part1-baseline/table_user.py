from sqlalchemy import Column, Integer, String
from sqlalchemy import func

from database import Base, SessionLocal


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    gender = Column(Integer)
    age = Column(Integer)
    country = Column(String)
    city = Column(String, )
    exp_group = Column(Integer)
    os = Column(String)
    source = Column(String)


if __name__ == "__main__":
    session = SessionLocal()

    try:
        # Выполнение запроса
        results = (session.query(User.country, User.os, func.count().label('count'))
                   .filter(User.exp_group == 3)
                   .group_by(User.country, User.os)
                   .having(func.count() > 100)
                   .order_by(func.count().desc())
                   .all()
                   )

        # Преобразование результатов в список кортежей
        result_list = [(country, os, count) for country, os, count in results]

        # Печать результата
        print(result_list)

    finally:
        session.close()
