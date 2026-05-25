# persistence/ — Chat Storage Framework

## Overview

Provides the chat persistence framework. The library ships the contract (`OrchidChatStorage` ABC), data models, migration runner, and **built-in SQLite (default)** backend. PostgreSQL backends are available via the `orchid-storage-postgres` plugin package. Consumers can override via dotted import paths.

## Architecture

```
orchid/persistence/                   ← LIBRARY (framework + built-in SQLite backend)
  base.py                               OrchidChatStorage ABC — the contract
  sqlite.py                             OrchidSQLiteChatStorage — built-in DEFAULT
  factory.py                            build_chat_storage(class_path, dsn) — dynamic import
  models.py                             OrchidChatSession, OrchidChatMessage — pure dataclasses

  mcp_token_sqlite.py                   OrchidSQLiteMCPTokenStore — per-user OAuth tokens
  mcp_token_factory.py                  build_mcp_token_store(class_path, dsn)

  mcp_client_registration_sqlite.py     OrchidSQLiteMCPClientRegistrationStore —
                                         per-server discovered endpoints + DCR
                                         credentials (MCP 2025-03-26 / RFC 7591)
  mcp_client_registration_factory.py    build_mcp_client_registration_store(class_path, dsn)

  migrations/
    runner.py                           OrchidMigrationRunner base + discover_migrations(package)
    v001_initial_schema.py              chat_sessions, chat_messages, mcp_oauth_tokens
    v002_mcp_client_registrations.py    mcp_client_registrations (MCP 2025-03-26 DCR)
```

Consumer projects can provide their own storage backends (e.g. PostgreSQL with custom migrations) by subclassing `OrchidChatStorage` and referencing via dotted import path.

## How It Works

The factory resolves a **dotted import path** to a `OrchidChatStorage` subclass at runtime:

```env
# PostgreSQL (install orchid-storage-postgres plugin first):
CHAT_STORAGE_CLASS=orchid_storage_postgres.OrchidPostgresChatStorage
CHAT_DB_DSN=postgresql://user:pass@host:5432/db

# SQLite (basketball example):
CHAT_STORAGE_CLASS=examples.basketball.storage.sqlite.OrchidSQLiteChatStorage
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

## OrchidChatStorage Interface

```python
class OrchidChatStorage(ABC):
    async def init_db(self) -> None          # open connection + run migrations
    async def close(self) -> None            # release connections
    async def create_chat(tenant_id, user_id, title) -> OrchidChatSession
    async def list_chats(tenant_id, user_id) -> list[OrchidChatSession]
    async def get_chat(chat_id) -> OrchidChatSession | None
    async def delete_chat(chat_id) -> None   # CASCADE deletes messages
    async def update_title(chat_id, title) -> None
    async def mark_shared(chat_id) -> None
    async def add_message(chat_id, role, content, agents_used, metadata) -> OrchidChatMessage
    async def get_messages(chat_id, limit, offset) -> list[OrchidChatMessage]
```

## Writing a Custom Backend

1. Create a Python file anywhere importable (e.g., `my_project/storage/mysql.py`)
2. Subclass `OrchidChatStorage` from `orchid.persistence.base`
3. Subclass `OrchidMigrationRunner` from `orchid.persistence.migrations.runner` — set `dialect` and `migrations_package`
4. Create migration modules in your migrations package
5. The constructor must accept `*, dsn: str, extra_migrations_package: str | None = None` and forward the extras kwarg to the migrator
6. Set `CHAT_STORAGE_CLASS=my_project.storage.mysql.MySQLChatStorage`

**Integrator migrations (recommended path for most consumers).** If you
only need extra tables/indices on top of the built-in PostgreSQL or
SQLite backend, don't subclass anything — point `storage.class` at the
framework backend and set `storage.extra_migrations_package` to the
dotted path of your migrations package:

```yaml
storage:
  class: orchid_storage_postgres.chat_storage.OrchidPostgresChatStorage
  dsn: postgresql://...
  extra_migrations_package: myapp.migrations
```

Framework migrations run first (recorded as `"001"`, `"002"`, …).
Integrator migrations run second, recorded with an `"ext:"` prefix
(`"ext:001"`, `"ext:002"`, …) — your file can start at `VERSION = "001"`
without colliding. The MCP OAuth token store reuses the same extras
package automatically (it shares the DB).

## Migration System

### OrchidMigrationRunner

```python
class OrchidMigrationRunner:
    dialect: str = "postgres"          # subclass sets this
    migrations_package: str | None     # framework package (subclass default)
    extra_migrations_package: str | None  # integrator (passed at construction)

    async def ensure_migrations_table(conn)
    async def get_applied_versions(conn) → set
    async def record_version(conn, version, description)
    async def remove_version(conn, version)
    async def run_up(conn)              # framework pass, then integrator pass
    async def run_down(conn, target_version)  # integrator first, then framework
```

The runner applies migrations in **two passes**:

1. Framework migrations from `self.migrations_package`, recorded with
   bare version keys (`"001"`, `"002"`, …).
2. Integrator migrations from `self.extra_migrations_package` (if set),
   recorded with the `"ext:"` prefix from
   `orchid_ai.persistence.migrations.runner.EXTRA_NAMESPACE_PREFIX`.

Rollback runs in reverse order (integrator first) to preserve
dependency direction.

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
- **`asyncpg` is NOT a dependency of orchid-ai** — install via `pip install orchid-storage-postgres` for the PostgreSQL backend.
- **Constructor signature:** All backends must accept `*, dsn: str` (keyword-only).
- **The factory uses `importlib`.** The class path must be importable from the working directory.
