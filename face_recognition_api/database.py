from sqlalchemy import create_engine, engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = 'sqlite:///./faces.db'

#the second argument (connect_args) only has to be passed when we are using SQLite database
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args= {
    "check_same_thread":False
})

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()