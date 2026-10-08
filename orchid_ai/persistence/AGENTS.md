# persistence/ — Chat Storage Framework

## Overview

Provides the storage persistence framework. The library ships the
contracts (`OrchidChatStorage` ABC, `OrchidConfigStorage`, the MCP store
ABCs), data models, dependency-free **in-memory defaults**, and the
migration-runner ABC. Durable backends are provided by plugin packages:

- `orchid-storage-sqlite` — chat, config, MCP stores, ingestion
  manifest, events backend/queue, migrations v001 + v002.
- `orchid-storage-postgres` — the same surface against PostgreSQL.

Consumers can also provide their own backends via dotted import paths.

## Architecture

```
orchid/persistence/                   ← LIBRARY (contracts + in-memory default)
  base.py                               OrchidChatStorage ABC — the contract
  models.py                             OrchidChatSession, OrchidChatMessage — pure dataclasses
  in_memory.py                          OrchidInMemory* stores — framework DEFAULT
  factory.py                            build_chat_storage(class_path, dsn) — dynamic import
  plugin_hints.py                       pip-install hints for plugin class paths
  mcp_token_factory.py                  build_mcp_token_store(class_path, dsn)
  mcp_client_registration_factory.py    build_mcp_client_registration_store(...)
  mcp_gateway_state_factory.py          build_mcp_gateway_state_store(...)

  migrations/
    runner.py                           OrchidMigrationRunner ABC + discover_migrations(package)
```

Concrete migrations live in the plugin packages
(`orchid_storage_sqlite.migrations`, `orchid_storage_postgres.migrations`)
or in consumer projects.

## How It Works

Storage class resolution has two modes:

1. **`"memory"` sentinel** (also the empty string) — selects the built-in
   in-memory backend. This is the framework default when no storage is
   configured; state is process-local and **not durable**.
2. **Dotted import path** — dynamically imported at runtime. A configured
   path always wins over the default and fails strictly when it cannot be
   imported (never silently falls back to in-memory). Known plugin
   prefixes get an actionable hint: `pip install orchid-storage-sqlite` /
   `pip install orchid-storage-postgres`.

```env
# In-memory (default — no configuration needed)
CHAT_STORAGE_CLASS=memory

# SQLite (install orchid-storage-sqlite):
CHAT_STORAGE_CLASS=orchid_storage_sqlite.chat_storage.OrchidSQLiteChatStorage
CHAT_DB_DSN=~/.orchid/chats.db

# PostgreSQL (install orchid-storage-postgres):
CHAT_STORAGE_CLASS=orchid_storage_postgres.OrchidPostgresChatStorage
CHAT_DB_DSN=postgresql://user:pass@host:5432/db
```

```python
# orchid_ai/persistence/factory.py
storage = build_chat_storage(
    class_path=settings.chat_storage_class,  # "memory" or a dotted path
    dsn=settings.chat_db_dsn,  # ignored by the in-memory backend
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

`get_chat_metadata` / `can_write` / `get_conversation_summary` /
`save_conversation_summary` have concrete ABC defaults so existing
backends keep working unchanged.

## Writing a Custom Backend

1. Create a Python file anywhere importable (e.g., `my_project/storage/mysql.py`)
2. Subclass `OrchidChatStorage` from `orchid_ai.persistence.base`
3. Subclass `OrchidMigrationRunner` from `orchid_ai.persistence.migrations.runner` — set `dialect` and `migrations_package`
4. Create migration modules in your migrations package
5. The constructor must accept `*, dsn: str, extra_migrations_package: str | None = None` and forward the extras kwarg to the migrator
6. Set `CHAT_STORAGE_CLASS=my_project.storage.mysql.MySQLChatStorage`

**Integrator migrations (recommended path for most consumers).** If you
only need extra tables/indices on top of a plugin backend, don't subclass
anything — point `storage.class` at the plugin backend and set
`storage.extra_migrations_package` to the dotted path of your migrations
package:

```yaml
storage:
  class: orchid_storage_sqlite.chat_storage.OrchidSQLiteChatStorage
  dsn: ~/.orchid/chats.db
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
    migrations_package: str | None     # backend package (subclass default)
    extra_migrations_package: str | None  # integrator (passed at construction)

    async def ensure_migrations_table(conn)
    async def get_applied_versions(conn) → set
    async def record_version(conn, version, description)
    async def remove_version(conn, version)
    async def run_up(conn)              # framework pass, then integrator pass
    async def run_down(conn, target_version)  # integrator first, then framework
```

The runner applies migrations in **two passes**:

1. Backend migrations from `self.migrations_package`, recorded with
   bare version keys (`"001"`, `"002"`, …).
2. Integrator migrations from `self.extra_migrations_package` (if set),
   recorded with the `"ext:"` prefix from
   `orchid_ai.persistence.migrations.runner.EXTRA_NAMESPACE_PREFIX`.

Rollback runs in reverse order (integrator first) to preserve
dependency direction.

### discover_migrations(package)

Scans the given package for modules starting with `v` that expose
`VERSION`, `up(conn, *, dialect)`, `down(conn, *, dialect)`. The
`package` parameter is a dotted import path (e.g.,
`"orchid_storage_sqlite.migrations"`). The framework package itself
carries no migration modules — a runner subclass always sets its
backend's package.

### Dialect-Aware Migrations

```python
async def up(conn, *, dialect: str = "sqlite") -> None:
    if dialect == "sqlite":
        await conn.execute("...")  # SQLite SQL
    else:
        await conn.execute("...")  # PostgreSQL SQL
```

## Important

- **No database driver is a core dependency.** In-memory defaults are
  stdlib-only; `aiosqlite` ships with `orchid-storage-sqlite`,
  `asyncpg` with `orchid-storage-postgres`.
- **Constructor signature:** All backends must accept `*, dsn: str`
  (keyword-only) plus `extra_migrations_package: str | None = None` for
  the SQL-backed ones.
- **The factory uses `importlib`.** The class path must be importable
  from the working directory.
- **Never silently fall back.** A configured-but-unimportable class path
  is a hard error (with the pip hint); only the unset/`"memory"` default
  selects the in-memory backend.
