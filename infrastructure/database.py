"""
Database infrastructure - SQLAlchemy engine, session factory, and ORM models
"""

from sqlalchemy import JSON, Column, DateTime, Integer, String, create_engine, func
from sqlalchemy.orm import declarative_base, sessionmaker

from config import config_data as config

Base = declarative_base()


class ChatMessageModel(Base):
    """Chat message record stored in MySQL"""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    content = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<ChatMessageModel(id={self.id}, session_id={self.session_id!r})>"


# Lazy-initialized engine and session factory
_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        db_url = (
            f"mysql+pymysql://{config.mysql_user}:{config.mysql_password}"
            f"@{config.mysql_host}:{config.mysql_port}/{config.mysql_database}"
            f"?charset=utf8mb4"
        )
        _engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_get_engine())
    return _SessionLocal


def init_db():
    """Initialize database tables. Call once at application startup."""
    engine = _get_engine()
    Base.metadata.create_all(engine)
