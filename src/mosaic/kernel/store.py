"""Append-only SQLite event store for the MOSAIC institutional kernel."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Iterator

from .events import Event, canonical_json
from .identity import AgentIdentity


class LedgerIntegrityError(RuntimeError):
    """Raised when the persisted event hash chain fails verification."""


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """An event plus ledger sequencing and integrity information."""

    sequence: int
    event: Event
    previous_hash: str | None
    event_hash: str


class SQLiteEventStore:
    """Small append-only event ledger backed by SQLite.

    Database triggers reject ordinary UPDATE and DELETE operations. Each row is
    additionally chained to the previous row with SHA-256, making accidental or
    unsophisticated tampering detectable by :meth:`verify_chain`.

    This is tamper-evident, not magically tamper-proof: an attacker with full
    database/schema access could remove triggers and recompute the chain.
    External checkpoints/signatures can be added later without changing the
    event envelope.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = RLock()
        self._conn = sqlite3.connect(
            self.path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                stream_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                occurred_at TEXT NOT NULL,

                actor_id TEXT NOT NULL,
                actor_version TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                actor_domains_json TEXT NOT NULL,

                payload_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,

                causation_id TEXT,
                correlation_id TEXT,

                previous_hash TEXT,
                event_hash TEXT NOT NULL UNIQUE
            );

            CREATE INDEX IF NOT EXISTS idx_events_stream_sequence
                ON events(stream_id, sequence);

            CREATE INDEX IF NOT EXISTS idx_events_type_sequence
                ON events(event_type, sequence);

            CREATE TRIGGER IF NOT EXISTS events_reject_update
            BEFORE UPDATE ON events
            BEGIN
                SELECT RAISE(ABORT, 'MOSAIC event ledger is append-only');
            END;

            CREATE TRIGGER IF NOT EXISTS events_reject_delete
            BEFORE DELETE ON events
            BEGIN
                SELECT RAISE(ABORT, 'MOSAIC event ledger is append-only');
            END;
            """
        )

    @staticmethod
    def _hash_material(event: Event, previous_hash: str | None) -> dict:
        return {
            "event_id": event.event_id,
            "stream_id": event.stream_id,
            "event_type": event.event_type,
            "schema_version": event.schema_version,
            "occurred_at": event.occurred_at.isoformat(),
            "actor": {
                "agent_id": event.actor.agent_id,
                "version": event.actor.version,
                "role": event.actor.role,
                "domains": list(event.actor.domains),
            },
            "payload": dict(event.payload),
            "metadata": dict(event.metadata),
            "causation_id": event.causation_id,
            "correlation_id": event.correlation_id,
            "previous_hash": previous_hash,
        }

    @classmethod
    def _calculate_hash(cls, event: Event, previous_hash: str | None) -> str:
        encoded = canonical_json(
            cls._hash_material(event, previous_hash)
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def append(self, event: Event) -> StoredEvent:
        """Atomically append one event to the global ledger."""

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                previous_row = self._conn.execute(
                    "SELECT event_hash FROM events "
                    "ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                previous_hash = (
                    previous_row["event_hash"] if previous_row else None
                )
                event_hash = self._calculate_hash(event, previous_hash)

                cursor = self._conn.execute(
                    """
                    INSERT INTO events (
                        event_id,
                        stream_id,
                        event_type,
                        schema_version,
                        occurred_at,
                        actor_id,
                        actor_version,
                        actor_role,
                        actor_domains_json,
                        payload_json,
                        metadata_json,
                        causation_id,
                        correlation_id,
                        previous_hash,
                        event_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.stream_id,
                        event.event_type,
                        event.schema_version,
                        event.occurred_at.isoformat(),
                        event.actor.agent_id,
                        event.actor.version,
                        event.actor.role,
                        canonical_json(list(event.actor.domains)),
                        canonical_json(dict(event.payload)),
                        canonical_json(dict(event.metadata)),
                        event.causation_id,
                        event.correlation_id,
                        previous_hash,
                        event_hash,
                    ),
                )
                sequence = int(cursor.lastrowid)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        return StoredEvent(
            sequence=sequence,
            event=event,
            previous_hash=previous_hash,
            event_hash=event_hash,
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> Event:
        identity = AgentIdentity(
            agent_id=row["actor_id"],
            version=row["actor_version"],
            role=row["actor_role"],
            domains=tuple(json.loads(row["actor_domains_json"])),
        )
        return Event(
            event_id=row["event_id"],
            stream_id=row["stream_id"],
            event_type=row["event_type"],
            schema_version=row["schema_version"],
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            actor=identity,
            payload=json.loads(row["payload_json"]),
            metadata=json.loads(row["metadata_json"]),
            causation_id=row["causation_id"],
            correlation_id=row["correlation_id"],
        )

    @classmethod
    def _stored_from_row(cls, row: sqlite3.Row) -> StoredEvent:
        return StoredEvent(
            sequence=row["sequence"],
            event=cls._event_from_row(row),
            previous_hash=row["previous_hash"],
            event_hash=row["event_hash"],
        )

    def get(self, event_id: str) -> StoredEvent | None:
        row = self._conn.execute(
            "SELECT * FROM events WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        return self._stored_from_row(row) if row else None

    def events_for_stream(self, stream_id: str) -> list[StoredEvent]:
        rows = self._conn.execute(
            "SELECT * FROM events "
            "WHERE stream_id = ? ORDER BY sequence",
            (stream_id,),
        ).fetchall()
        return [self._stored_from_row(row) for row in rows]

    def iter_all(self) -> Iterator[StoredEvent]:
        rows = self._conn.execute(
            "SELECT * FROM events ORDER BY sequence"
        ).fetchall()
        for row in rows:
            yield self._stored_from_row(row)

    def verify_chain(self) -> bool:
        """Return True only if every stored event matches its chain hash."""

        expected_previous: str | None = None
        for stored in self.iter_all():
            if stored.previous_hash != expected_previous:
                return False

            expected_hash = self._calculate_hash(
                stored.event,
                expected_previous,
            )
            if stored.event_hash != expected_hash:
                return False

            expected_previous = stored.event_hash

        return True

    def assert_integrity(self) -> None:
        if not self.verify_chain():
            raise LedgerIntegrityError(
                "MOSAIC event ledger hash-chain verification failed"
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteEventStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
