from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .feedback import FeedbackWeights
from .interactive import InteractiveSession, SearchIndex, SessionTransition


SCHEMA_VERSION = 1


class SessionStoreError(RuntimeError):
    pass


class StoredSessionNotFound(SessionStoreError):
    pass


class StoredSessionAlreadyExists(SessionStoreError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _vector_blob(vector: np.ndarray | Sequence[float] | None) -> bytes | None:
    if vector is None:
        return None
    return np.ascontiguousarray(np.asarray(vector, dtype=np.float32)).tobytes()


def _blob_vector(value: bytes | None, dimension: int) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.frombuffer(value, dtype=np.float32).copy()
    if vector.shape != (dimension,):
        raise SessionStoreError("stored vector dimension is incompatible")
    return vector


def _request_digest(
    shown: Sequence[str],
    accepted: Sequence[str],
    rejected: Sequence[str],
    feedback_text_embedding: np.ndarray | Sequence[float] | None,
) -> str:
    blob = _vector_blob(feedback_text_embedding)
    payload = {
        "shown": list(shown),
        "accepted": sorted(accepted),
        "rejected": sorted(rejected),
        "feedback_embedding_sha256": hashlib.sha256(blob).hexdigest()
        if blob is not None
        else None,
    }
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


class SQLiteSessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    reference_image_id TEXT NOT NULL,
                    initial_query BLOB NOT NULL,
                    encoder_version TEXT NOT NULL,
                    index_version TEXT NOT NULL,
                    reranker_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    feedback_weights_json TEXT NOT NULL,
                    max_rounds INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    close_reason TEXT,
                    last_shown_json TEXT NOT NULL,
                    last_shown_round INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback_events (
                    session_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    shown_json TEXT NOT NULL,
                    accepted_json TEXT NOT NULL,
                    rejected_json TEXT NOT NULL,
                    feedback_text_embedding BLOB,
                    request_digest TEXT NOT NULL,
                    transition_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, event_id),
                    UNIQUE (session_id, sequence),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                );
                """
            )
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row["value"]) != SCHEMA_VERSION:
                raise SessionStoreError(
                    f"unsupported session database schema: {row['value']}"
                )

    def create_session(self, session: InteractiveSession) -> None:
        snapshot = session.snapshot()
        now = _utc_now()
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO sessions(
                        session_id, reference_image_id, initial_query,
                        encoder_version, index_version, reranker_version,
                        prompt_version, feedback_weights_json, max_rounds,
                        status, close_reason, last_shown_json, last_shown_round,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.session_id,
                        snapshot.reference_image_id,
                        _vector_blob(session.initial_query),
                        snapshot.encoder_version,
                        snapshot.index_version,
                        snapshot.reranker_version,
                        snapshot.prompt_version,
                        _json(asdict(session.weights)),
                        snapshot.max_rounds,
                        snapshot.status,
                        snapshot.close_reason,
                        _json(snapshot.last_shown_image_ids),
                        None,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise StoredSessionAlreadyExists(snapshot.session_id) from exc

    def _update_snapshot(
        self, connection: sqlite3.Connection, session: InteractiveSession
    ) -> None:
        snapshot = session.snapshot()
        result = connection.execute(
            """
            UPDATE sessions SET
                status = ?, close_reason = ?, last_shown_json = ?,
                last_shown_round = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (
                snapshot.status,
                snapshot.close_reason,
                _json(snapshot.last_shown_image_ids),
                snapshot.round_id if snapshot.last_shown_image_ids else None,
                _utc_now(),
                snapshot.session_id,
            ),
        )
        if result.rowcount != 1:
            raise StoredSessionNotFound(snapshot.session_id)

    def record_exposure(self, session: InteractiveSession) -> None:
        with closing(self._connect()) as connection, connection:
            self._update_snapshot(connection, session)

    def record_feedback(
        self,
        session: InteractiveSession,
        *,
        shown_image_ids: Sequence[str],
        accepted_ids: Sequence[str],
        rejected_ids: Sequence[str],
        feedback_text_embedding: np.ndarray | Sequence[float] | None,
        transition: SessionTransition,
    ) -> None:
        digest = _request_digest(
            shown_image_ids,
            accepted_ids,
            rejected_ids,
            feedback_text_embedding,
        )
        with closing(self._connect()) as connection, connection:
            existing = connection.execute(
                """
                SELECT request_digest FROM feedback_events
                WHERE session_id = ? AND event_id = ?
                """,
                (session.session_id, transition.event_id),
            ).fetchone()
            if existing is not None:
                if existing["request_digest"] != digest:
                    raise SessionStoreError(
                        "stored event_id has different feedback content"
                    )
                return
            connection.execute(
                """
                INSERT INTO feedback_events(
                    session_id, event_id, sequence, shown_json, accepted_json,
                    rejected_json, feedback_text_embedding, request_digest,
                    transition_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    transition.event_id,
                    transition.round_after,
                    _json(list(shown_image_ids)),
                    _json(sorted(set(accepted_ids))),
                    _json(sorted(set(rejected_ids))),
                    _vector_blob(feedback_text_embedding),
                    digest,
                    _json(transition.to_dict()),
                    _utc_now(),
                ),
            )
            self._update_snapshot(connection, session)

    def record_close(self, session: InteractiveSession) -> None:
        with closing(self._connect()) as connection, connection:
            self._update_snapshot(connection, session)

    def load_session(
        self,
        session_id: str,
        *,
        index: SearchIndex,
        embedding_by_id: Mapping[str, np.ndarray],
    ) -> InteractiveSession:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise StoredSessionNotFound(session_id)
            events = connection.execute(
                """
                SELECT * FROM feedback_events WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()

        if row["index_version"] != index.manifest.index_version:
            raise SessionStoreError("stored session index version is unavailable")
        if row["encoder_version"] != index.manifest.encoder_version:
            raise SessionStoreError("stored session encoder version is unavailable")
        dimension = index.manifest.embedding_dim
        initial = _blob_vector(row["initial_query"], dimension)
        assert initial is not None
        session = InteractiveSession(
            session_id=row["session_id"],
            reference_image_id=row["reference_image_id"],
            initial_query=initial,
            encoder_version=row["encoder_version"],
            index=index,
            embedding_by_id=embedding_by_id,
            weights=FeedbackWeights(**json.loads(row["feedback_weights_json"])),
            max_rounds=int(row["max_rounds"]),
            reranker_version=row["reranker_version"],
            prompt_version=row["prompt_version"],
        )
        for event in events:
            shown = tuple(json.loads(event["shown_json"]))
            actual = tuple(item.image_id for item in session.retrieve(len(shown)))
            if actual != shown:
                raise SessionStoreError(
                    "stored exposure cannot be reproduced with the loaded index"
                )
            transition = session.submit_feedback(
                event_id=event["event_id"],
                shown_image_ids=shown,
                accepted_ids=tuple(json.loads(event["accepted_json"])),
                rejected_ids=tuple(json.loads(event["rejected_json"])),
                feedback_text_embedding=_blob_vector(
                    event["feedback_text_embedding"], dimension
                ),
            )
            persisted = json.loads(event["transition_json"])
            if _json(transition.to_dict()) != _json(persisted):
                raise SessionStoreError("replayed transition differs from stored event")

        last_shown = tuple(json.loads(row["last_shown_json"]))
        if last_shown:
            actual = tuple(item.image_id for item in session.retrieve(len(last_shown)))
            if actual != last_shown:
                raise SessionStoreError("last stored exposure cannot be reproduced")
        stored_status = row["status"]
        if stored_status == "closed" and session.snapshot().status == "active":
            session.close(row["close_reason"] or "user_closed")
        if session.snapshot().status != stored_status:
            raise SessionStoreError("replayed session status differs from stored status")
        return session

    def event_count(self, session_id: str) -> int:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM feedback_events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["count"])

    def session_index_version(self, session_id: str) -> str:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT index_version FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise StoredSessionNotFound(session_id)
        return str(row["index_version"])
