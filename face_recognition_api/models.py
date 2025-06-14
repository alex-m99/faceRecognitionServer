from sqlalchemy import Column, Integer, String, Text
from .database import Base
import json

# database model
class Person(Base):
    __tablename__ = 'people'
    id = Column(Integer, primary_key=True, index=True)
    firstName = Column(String)
    lastName = Column(String)
    function = Column(String)
    email = Column(String)
    encoding = Column("encoding", Text)

    # @property
    # def encoding(self) -> list[float]:
    #     return json.loads(self._encoding)

    # @encoding.setter
    # def encoding(self, value: list[float]):
    #     self._encoding = json.dumps(value)
