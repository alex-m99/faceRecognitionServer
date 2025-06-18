from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Table
from sqlalchemy.orm import relationship
from .database import Base
import datetime

class Admin(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)

class System(Base):
    __tablename__ = 'systems'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    password = Column(String, nullable=False)
    lock_password = Column(String, nullable=False)
    address = Column(String, nullable=False)
    started = Column(Boolean, default=True)
    starting_date = Column(DateTime, default=datetime.datetime.utcnow)
    system_token = Column(String, nullable=True)  # <-- Add this
    

    # Relationship to PersonSystem
    persons = relationship("PersonSystem", back_populates="system")

class PersonSystem(Base):
    __tablename__ = 'person_system'
    person_id = Column(Integer, ForeignKey('people.id'), primary_key=True)
    system_id = Column(Integer, ForeignKey('systems.id'), primary_key=True)
    access = Column(Boolean, default=True)

    # Relationships
    person = relationship("Person", back_populates="systems")
    system = relationship("System", back_populates="persons")

# Update Person to add relationship
class Person(Base):
    __tablename__ = 'people'
    id = Column(Integer, primary_key=True, index=True)
    firstName = Column(String)
    lastName = Column(String)
    function = Column(String)
    email = Column(String)
    photo_url = Column(String, nullable=True)
    encoding = Column("encoding", Text)

    # Relationship to PersonSystem
    systems = relationship("PersonSystem", back_populates="person")