"""Deterministic end-to-end smoke test for the multi-turn session core."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cir_baseline.contracts import IndexManifest  # noqa: E402
from cir_baseline.index import NumpyFlatIPIndex  # noqa: E402
from cir_baseline.interactive import InteractiveSession  # noqa: E402


def main() -> int:
    ids = ["reference", "candidate-a", "candidate-b", "candidate-c"]
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.2, 0.9, 0.3],
            [0.0, 0.2, 1.0],
        ],
        dtype=np.float32,
    )
    manifest = IndexManifest(
        index_version="interactive-smoke-flatip-v1",
        encoder_version="interactive-smoke-encoder-v1",
        embedding_dim=3,
    )
    index = NumpyFlatIPIndex(ids, embeddings, manifest)
    session = InteractiveSession(
        session_id="anon-smoke-session",
        reference_image_id="reference",
        initial_query=np.asarray([0.7, 0.7, 0.0], dtype=np.float32),
        encoder_version=manifest.encoder_version,
        index=index,
        embedding_by_id=dict(zip(ids, embeddings, strict=True)),
        max_rounds=2,
    )

    rounds = []
    first_results = session.retrieve(top_k=3)
    first_shown = tuple(item.image_id for item in first_results)
    first = session.submit_feedback(
        event_id="smoke-event-1",
        shown_image_ids=first_shown,
        accepted_ids=("candidate-b",),
        rejected_ids=("candidate-c",),
    )
    repeated = session.submit_feedback(
        event_id="smoke-event-1",
        shown_image_ids=first_shown,
        accepted_ids=("candidate-b",),
        rejected_ids=("candidate-c",),
    )
    rounds.append(
        {
            "shown": first_shown,
            "transition": first.to_dict(),
            "idempotent_retry_equal": repeated == first,
        }
    )

    second_results = session.retrieve(top_k=3)
    second_shown = tuple(item.image_id for item in second_results)
    second = session.submit_feedback(
        event_id="smoke-event-2",
        shown_image_ids=second_shown,
        accepted_ids=("candidate-a",),
        feedback_text_embedding=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    rounds.append({"shown": second_shown, "transition": second.to_dict()})

    snapshot = session.snapshot()
    checks = {
        "reference_excluded_from_every_exposure": all(
            "reference" not in row["shown"] for row in rounds
        ),
        "rounds_incremented_monotonically": [
            row["transition"]["round_after"] for row in rounds
        ]
        == [1, 2],
        "idempotent_retry_did_not_add_round": rounds[0][
            "idempotent_retry_equal"
        ],
        "explicit_negative_retained": snapshot.negative_image_ids
        == ("candidate-c",),
        "latest_positive_labels_retained": snapshot.positive_image_ids
        == ("candidate-a", "candidate-b"),
        "max_rounds_closed_session": snapshot.status == "completed"
        and snapshot.close_reason == "max_rounds_reached",
        "current_query_is_unit_normalized": abs(
            float(np.linalg.norm(session.current_query)) - 1.0
        )
        < 1e-6,
    }
    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "synthetic multi-turn state-machine smoke test; not a research result",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "rounds": rounds,
        "final_snapshot": snapshot.to_dict(),
    }
    target = PROJECT_ROOT / "artifacts" / "metrics" / "stage3_interactive_smoke.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
