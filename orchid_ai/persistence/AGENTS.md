# persistence/ — Chat Storage Framework

## Overview

Provides the chat persistence framework. The library ships the contract (`ChatStorage` ABC), data models, migration runner, **built-in SQLite (default)** and **PostgreSQL** backends. Consumers can override via dotted import paths.

## Architecture

```
orchid/persistence/                   ← LIBRARY (framework + built-in backends)
  base.py                               ChatStorage ABC — the contract
  sqlite.py                             SQLiteChatStorage — built-in DEFAULT
  postgres.py                           PostgresChatStorage — built-in (requires asyncpg)
  factory.py                            build_chat_storage(class_path, dsn) — dynamic import
  models.py                             ChatSession, ChatMessage — pure dataclasses
  migrations/
    runner.py                           MigrationRunner base + discover_migrations(package)
    v001_initial_schema.py              Dialect-aware schema (PG + SQLite)

```

Consumer projects can provide their own storage backends (e.g. PostgreSQL with custom migrations) by subclassing `ChatStorage` and referencing via dotted import path.

## How It Works

The factory resolves a **dotted import path** to a `ChatStorage` subclass at runtime:

```env
# Built-in PostgreSQL (default):
CHAT_STORAGE_CLASS=orchid.persistence.postgres.PostgresChatStorage
CHAT_DB_DSN=postgresql://user:pass@host:5432/db

# SQLite (basketball example):
CHAT_STORAGE_CLASS=examples.basketball.storage.sqlite.SQLiteChatStorage
CHAT_DB_DSN=/data/chats.db
```

```python
# src/persistence/factory.py
storage = build_chat_storage(
    class_path=settings.chat_storage_class,
    dsn=settings.chat_db_dsn,
)
await storage.init_db()
```

## ChatStorage Interface

```python
class ChatStorage(ABC):
    async def init_db(self) -> None          # open connection + run migrations
    async def close(self) -> None            # release connections
    async def create_chat(tenant_id, user_id, title) -> ChatSession
    async def list_chats(tenant_id, user_id) -> list[ChatSession]
    async def get_chat(chat_id) -> ChatSession | None
    async def delete_chat(chat_id) -> None   # CASCADE deletes messages
    async def update_title(chat_id, title) -> None
    async def mark_shared(chat_id) -> None
    async def add_message(chat_id, role, content, agents_used, metadata) -> ChatMessage
    async def get_messages(chat_id, limit, offset) -> list[ChatMessage]
```

## Writing a Custom Backend

1. Create a Python file anywhere importable (e.g., `my_project/storage/mysql.py`)
2. Subclass `ChatStorage` from `orchid.persistence.base`
3. Subclass `MigrationRunner` from `orchid.persistence.migrations.runner` — set `dialect` and `migrations_package`
4. Create migration modules in your migrations package
5. The constructor must accept `*, dsn: str`
6. Set `CHAT_STORAGE_CLASS=my_project.storage.mysql.MySQLChatStorage`

## Migration System

### MigrationRunner

```python
class MigrationRunner:
    dialect: str = "postgres"          # subclass sets this
    migrations_package: str | None     # dotted path to migrations package

    async def ensure_migrations_table(conn)
    async def get_applied_versions(conn) → set
    async def record_version(conn, version, description)
    async def remove_version(conn, version)
    async def run_up(conn)
    async def run_down(conn, target_version)
```

### discover_migrations(package)

Scans the given package for modules starting with `v` that expose `VERSION`, `up(conn, *, dialect)`, `down(conn, *, dialect)`. The `package` parameter is a dotted import path (e.g., `"orchid.persistence.migrations"`).

### Dialect-Aware Migrations

```python
async def up(conn, *, dialect: str = "postgres") -> None:
    if dialect == "sqlite":
        await conn.execute("...")  # SQLite SQL
    else:
        await conn.execute("...")  # PostgreSQL SQL
```

## Important

- **`aiosqlite` is a core dependency** — it ships with the library for the built-in SQLite default backend.
- **`asyncpg` is an optional dependency** — install via `pip install "orchid[postgres]"` for the PostgreSQL backend.
- **Constructor signature:** All backends must accept `*, dsn: str` (keyword-only).
- **The factory uses `importlib`.** The class path must be importable from the working directory.
