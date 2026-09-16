"""SQLite 存储：对话历史、议价计数、商品缓存、会话画像。

这是"对账簿"——回答"到底发生过什么"，只追加、永不删改。
与之相对，session.py 的 JSONL 快照是"行车记录仪"，回答"模型当时看到了什么"。

能力层全是同步 sqlite3/requests，async 包装统一走 asyncio.to_thread，
绝不卡事件循环——一个买家触发一次慢查询，不能让所有会话一起停摆。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class Store:
    def __init__(self, db_path: str = "data/chat_history.db", max_history: int = 100):
        self.max_history = max_history
        self.db_path = db_path
        self._init_db()

    # ---------------------------------------------------------------- 底层（同步）

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            chat_id TEXT
        )""")

        cursor.execute("PRAGMA table_info(messages)")
        columns = [column[1] for column in cursor.fetchall()]
        if "chat_id" not in columns:
            cursor.execute("ALTER TABLE messages ADD COLUMN chat_id TEXT")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_item ON messages (user_id, item_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_id ON messages (chat_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON messages (timestamp)")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_bargain_counts (
            chat_id TEXT PRIMARY KEY,
            count INTEGER DEFAULT 0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            item_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            price REAL,
            description TEXT,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_profiles (
            chat_id TEXT PRIMARY KEY,
            profile TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")

        conn.commit()
        conn.close()
        logger.info("聊天历史数据库初始化完成: %s", self.db_path)

    def _add_message(self, chat_id, user_id, item_id, role, content) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO messages (user_id, item_id, role, content, timestamp, chat_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, item_id, role, content, datetime.now().isoformat(), chat_id),
            )
            # 清理超出上限的旧消息（原项目逻辑）
            cursor = conn.execute(
                "SELECT id FROM messages WHERE chat_id = ? "
                "ORDER BY timestamp DESC LIMIT ?, 1",
                (chat_id, self.max_history),
            )
            oldest_to_keep = cursor.fetchone()
            if oldest_to_keep:
                conn.execute("DELETE FROM messages WHERE chat_id = ? AND id < ?",
                             (chat_id, oldest_to_keep[0]))
            conn.commit()
        except Exception:
            logger.exception("添加消息到数据库时出错")
            conn.rollback()
        finally:
            conn.close()

    def _get_context(self, chat_id: str, limit: int | None) -> list[dict]:
        """取一个会话最近 limit 条消息（时间正序）。limit=None 取全部。"""
        conn = self._connect()
        try:
            if limit is None:
                cursor = conn.execute(
                    "SELECT role, content FROM messages WHERE chat_id = ? "
                    "ORDER BY timestamp ASC LIMIT ?",
                    (chat_id, self.max_history),
                )
            else:
                cursor = conn.execute(
                    "SELECT role, content FROM ("
                    "  SELECT role, content, timestamp FROM messages WHERE chat_id = ? "
                    "  ORDER BY timestamp DESC LIMIT ?"
                    ") ORDER BY timestamp ASC",
                    (chat_id, limit),
                )
            return [{"role": role, "content": content} for role, content in cursor.fetchall()]
        except Exception:
            logger.exception("获取对话历史时出错")
            return []
        finally:
            conn.close()

    def _increment_bargain_count(self, chat_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO chat_bargain_counts (chat_id, count, last_updated) "
                "VALUES (?, 1, ?) ON CONFLICT(chat_id) "
                "DO UPDATE SET count = count + 1, last_updated = ?",
                (chat_id, datetime.now().isoformat(), datetime.now().isoformat()),
            )
            conn.commit()
        except Exception:
            logger.exception("增加议价次数时出错")
            conn.rollback()
        finally:
            conn.close()

    def _get_bargain_count(self, chat_id: str) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT count FROM chat_bargain_counts WHERE chat_id = ?", (chat_id,))
            result = cursor.fetchone()
            return result[0] if result else 0
        except Exception:
            logger.exception("获取议价次数时出错")
            return 0
        finally:
            conn.close()

    def _save_item_info(self, item_id, item_data) -> None:
        conn = self._connect()
        try:
            price = float(item_data.get("soldPrice", 0) or 0)
            description = item_data.get("desc", "")
            data_json = json.dumps(item_data, ensure_ascii=False)
            now = datetime.now().isoformat()
            conn.execute(
                "INSERT INTO items (item_id, data, price, description, last_updated) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(item_id) "
                "DO UPDATE SET data = ?, price = ?, description = ?, last_updated = ?",
                (item_id, data_json, price, description, now,
                 data_json, price, description, now),
            )
            conn.commit()
        except Exception:
            logger.exception("保存商品信息时出错")
            conn.rollback()
        finally:
            conn.close()

    def _get_item_info(self, item_id: str) -> dict | None:
        conn = self._connect()
        try:
            cursor = conn.execute("SELECT data FROM items WHERE item_id = ?", (item_id,))
            result = cursor.fetchone()
            return json.loads(result[0]) if result else None
        except Exception:
            logger.exception("获取商品信息时出错")
            return None
        finally:
            conn.close()

    def _get_profile(self, chat_id: str):
        """返回 (画像文本, 更新时间 epoch)。"""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT profile, updated_at FROM session_profiles WHERE chat_id = ?",
                (chat_id,))
            row = cursor.fetchone()
            if not row:
                return None, 0.0
            try:
                updated = datetime.fromisoformat(row[1]).timestamp()
            except (ValueError, TypeError):
                updated = 0.0
            return row[0], updated
        except Exception:
            logger.exception("获取会话画像时出错")
            return None, 0.0
        finally:
            conn.close()

    def _set_profile(self, chat_id: str, profile: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO session_profiles (chat_id, profile, updated_at) "
                "VALUES (?, ?, ?) ON CONFLICT(chat_id) "
                "DO UPDATE SET profile = ?, updated_at = ?",
                (chat_id, profile, datetime.now().isoformat(),
                 profile, datetime.now().isoformat()),
            )
            conn.commit()
        except Exception:
            logger.exception("保存会话画像时出错")
            conn.rollback()
        finally:
            conn.close()

    def _last_activity(self, chat_id: str) -> float:
        """会话最后一条消息的时间（epoch）。没有消息返回 0。"""
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT timestamp FROM messages WHERE chat_id = ? "
                "ORDER BY timestamp DESC LIMIT 1", (chat_id,))
            row = cursor.fetchone()
            if not row:
                return 0.0
            try:
                return datetime.fromisoformat(row[0]).timestamp()
            except (ValueError, TypeError):
                return 0.0
        except Exception:
            logger.exception("获取会话活跃时间时出错")
            return 0.0
        finally:
            conn.close()

    # ---------------------------------------------------------------- async 包装

    async def add_message(self, chat_id, user_id, item_id, role, content):
        return await asyncio.to_thread(self._add_message, chat_id, user_id, item_id, role, content)

    async def get_context_by_chat(self, chat_id: str, limit: int | None = None) -> list[dict]:
        return await asyncio.to_thread(self._get_context, chat_id, limit)

    async def increment_bargain_count(self, chat_id: str):
        return await asyncio.to_thread(self._increment_bargain_count, chat_id)

    async def get_bargain_count(self, chat_id: str) -> int:
        return await asyncio.to_thread(self._get_bargain_count, chat_id)

    async def save_item_info(self, item_id, item_data):
        return await asyncio.to_thread(self._save_item_info, item_id, item_data)

    async def get_item_info(self, item_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_item_info, item_id)

    async def get_profile(self, chat_id: str):
        return await asyncio.to_thread(self._get_profile, chat_id)

    async def set_profile(self, chat_id: str, profile: str):
        return await asyncio.to_thread(self._set_profile, chat_id, profile)

    async def last_activity(self, chat_id: str) -> float:
        return await asyncio.to_thread(self._last_activity, chat_id)
