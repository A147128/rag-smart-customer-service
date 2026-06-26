"""
MySQL chat message history store - replaces the old JSON file-based store
"""

from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from sqlalchemy import select

from infrastructure.database import ChatMessageModel, get_session_factory

_session_factory = None


def get_session_factory_singleton():
    global _session_factory
    if _session_factory is None:
        _session_factory = get_session_factory()
    return _session_factory


def get_history(session_id: str) -> "MySQLChatMessageHistory":
    return MySQLChatMessageHistory(session_id, get_session_factory_singleton())


class MySQLChatMessageHistory(BaseChatMessageHistory):
    """Chat message history backed by MySQL."""

    def __init__(self, session_id: str, session_factory) -> None:
        self.session_id = session_id
        self._session_factory = session_factory

    @property
    def messages(self) -> list[BaseMessage]:
        with self._session_factory() as session:
            stmt = (
                select(ChatMessageModel)
                .where(ChatMessageModel.session_id == self.session_id)
                .order_by(ChatMessageModel.id.asc())
            )
            records = session.scalars(stmt).all()
            dicts = [r.content for r in records]
            return messages_from_dict(dicts)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        with self._session_factory() as session:
            for msg in messages:
                session.add(
                    ChatMessageModel(
                        session_id=self.session_id,
                        content=message_to_dict(msg),
                    )
                )
            session.commit()

    def clear(self) -> None:
        with self._session_factory() as session:
            stmt = select(ChatMessageModel).where(ChatMessageModel.session_id == self.session_id)
            records = session.scalars(stmt).all()
            for record in records:
                session.delete(record)
            session.commit()
