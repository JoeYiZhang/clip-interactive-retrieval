from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


def _required_text(payload: Mapping[str, Any], name: str) -> str:
    value = str(payload.get(name, "")).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    values = tuple(str(item) for item in value)
    if any(not item for item in values):
        raise ValueError(f"{name} cannot contain empty values")
    return values


def _float_tuple(
    value: Any, name: str, *, optional: bool = False
) -> tuple[float, ...] | None:
    if value is None and optional:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of numbers")
    values = tuple(float(item) for item in value)
    if not values:
        raise ValueError(f"{name} must be non-empty")
    return values


@dataclass(frozen=True)
class CreateSessionRequest:
    session_id: str
    reference_image_id: str
    initial_query: tuple[float, ...]
    expected_encoder_version: str
    expected_index_version: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CreateSessionRequest":
        initial = _float_tuple(payload.get("initial_query"), "initial_query")
        assert initial is not None
        return cls(
            session_id=_required_text(payload, "session_id"),
            reference_image_id=_required_text(payload, "reference_image_id"),
            initial_query=initial,
            expected_encoder_version=_required_text(
                payload, "expected_encoder_version"
            ),
            expected_index_version=_required_text(payload, "expected_index_version"),
        )


@dataclass(frozen=True)
class RetrieveRequest:
    top_k: int = 10

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RetrieveRequest":
        top_k = int(payload.get("top_k", 10))
        if top_k <= 0 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        return cls(top_k=top_k)


@dataclass(frozen=True)
class FeedbackRequest:
    event_id: str
    shown_image_ids: tuple[str, ...]
    accepted_ids: tuple[str, ...] = ()
    rejected_ids: tuple[str, ...] = ()
    feedback_text_embedding: tuple[float, ...] | None = None
    expected_round_id: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeedbackRequest":
        raw_round = payload.get("expected_round_id")
        expected_round = None if raw_round is None else int(raw_round)
        if expected_round is not None and expected_round < 0:
            raise ValueError("expected_round_id must be non-negative")
        return cls(
            event_id=_required_text(payload, "event_id"),
            shown_image_ids=_string_tuple(
                payload.get("shown_image_ids"), "shown_image_ids"
            ),
            accepted_ids=_string_tuple(payload.get("accepted_ids", ()), "accepted_ids"),
            rejected_ids=_string_tuple(payload.get("rejected_ids", ()), "rejected_ids"),
            feedback_text_embedding=_float_tuple(
                payload.get("feedback_text_embedding"),
                "feedback_text_embedding",
                optional=True,
            ),
            expected_round_id=expected_round,
        )


@dataclass(frozen=True)
class CloseSessionRequest:
    reason: str = "user_closed"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CloseSessionRequest":
        reason = str(payload.get("reason", "user_closed")).strip()
        return cls(reason=reason or "user_closed")


@dataclass(frozen=True)
class RetrievalResponse:
    session: dict
    results: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {"session": self.session, "results": list(self.results)}


@dataclass(frozen=True)
class FeedbackResponse:
    session: dict
    transition: dict

    def to_dict(self) -> dict:
        return asdict(self)
