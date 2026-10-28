"""
app/db/session.py
==================
SQLAlchemy async-compatible session factory.

Usage
-----
    from app.db.session import get_session, init_db

    # On startup (FastAPI lifespan):
    await init_db()

    # In a request / node:
    with get_session() as session:
        session.add(record)
        session.commit()
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.models import Base

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        try:
            # Try configured database (Postgres) with 2-second timeout
            connect_args = {"connect_timeout": 2} if "psycopg" in settings.database_url else {}
            engine = create_engine(
                settings.database_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                connect_args=connect_args,
                echo=False,
            )
            with engine.connect() as test_conn:
                test_conn.execute(text("SELECT 1"))
            _engine = engine
            logger.info("Connected to primary database: %s", settings.database_url.split("@")[-1])
        except Exception as pg_err:
            sqlite_path = Path(settings.storage_dir) / "invoices.sqlite3"
            logger.warning("Primary database unavailable (%s). Falling back to local SQLite: %s", pg_err, sqlite_path)
            _engine = create_engine(
                f"sqlite:///{sqlite_path}",
                connect_args={"check_same_thread": False},
                echo=False,
            )
    return _engine


def _get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=_get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


def init_db() -> None:
    """Create all tables (idempotent). Called on application startup."""
    engine = _get_engine()
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
                logger.info("pgvector extension enabled")
            except Exception as exc:
                logger.warning("pgvector extension not available: %s", exc)
                try:
                    conn.rollback()
                except Exception:
                    pass

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified on dialect: %s", engine.dialect.name)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager providing a transactional DB session."""
    factory = _get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
