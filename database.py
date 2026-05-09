from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class VocabEntry(Base):
    __tablename__ = 'vocab_bank'
    id = Column(Integer, primary_key=True)
    kanji = Column(String(255))
    romaji = Column(String(255))
    meaning = Column(Text)
    source_video = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    engine = create_engine('sqlite:///immersion_data.db')
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()