"""Run one MOSAIC investigation against an OpenAI-compatible model server."""

from __future__ import annotations

import asyncio
import os
import sys

from mosaic import AgentIdentity, Investigation, SQLiteEventStore
from mosaic.agents import (
    ModelCoordinator,
    ModelCrossDomainReviewer,
    ModelExaminer,
    ModelThinker,
    ResearchCycle,
)
from mosaic.backends import OpenAICompatibleBackend


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} must be set")
    return value


async def main(user_input: str) -> None:
    model = require_env("MOSAIC_MODEL")
    base_url = os.getenv(
        "MOSAIC_BASE_URL",
        "https://api.openai.com/v1",
    )
    api_key = os.getenv("MOSAIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    database = os.getenv("MOSAIC_DB", "mosaic.db")

    backend = OpenAICompatibleBackend(
        backend_id="primary",
        base_url=base_url,
        api_key=api_key,
        api_key_env=None,
        structured_outputs=True,
    )

    coordinator_id = AgentIdentity(
        "coordinator",
        "v1",
        "coordinator",
    )
    examiner_id = AgentIdentity(
        "examiner",
        "v1",
        "examiner",
        ("analysis", "falsification"),
    )
    reviewer_id = AgentIdentity(
        "cross-domain-reviewer",
        "v1",
        "cross-domain-reviewer",
        ("systems", "adversarial-review"),
    )

    thinker_specs = (
        (
            AgentIdentity(
                "mechanical",
                "v1",
                "thinker",
                ("mechanical", "fluids", "thermodynamics"),
            ),
            "Mechanical systems, fuel delivery, fluids, mechanisms, materials.",
        ),
        (
            AgentIdentity(
                "electrical-controls",
                "v1",
                "thinker",
                ("electrical", "controls", "instrumentation"),
            ),
            "Electrical systems, controls, sensors, actuators, signal integrity.",
        ),
        (
            AgentIdentity(
                "physics",
                "v1",
                "thinker",
                ("physics", "first-principles"),
            ),
            "First-principles physics, dimensional reasoning, physical constraints.",
        ),
        (
            AgentIdentity(
                "experimental",
                "v1",
                "thinker",
                ("statistics", "experimental-design"),
            ),
            "Statistics, uncertainty, experimental design, confounding variables.",
        ),
    )

    thinkers = {
        identity.ref: ModelThinker(
            identity=identity,
            backend=backend,
            model=model,
            instructions=description,
        )
        for identity, description in thinker_specs
    }

    coordinator = ModelCoordinator(
        identity=coordinator_id,
        backend=backend,
        model=model,
        thinker_catalog={
            identity.ref: description
            for identity, description in thinker_specs
        },
    )
    examiner = ModelExaminer(
        identity=examiner_id,
        backend=backend,
        model=model,
    )
    reviewer = ModelCrossDomainReviewer(
        identity=reviewer_id,
        backend=backend,
        model=model,
    )

    cycle = ResearchCycle(
        coordinator=coordinator,
        thinkers=thinkers,
        examiner=examiner,
        reviewer=reviewer,
    )

    with SQLiteEventStore(database) as store:
        investigation = Investigation(store, "interactive")
        result = await cycle.run(
            investigation=investigation,
            user_input=user_input,
        )

    print(result.report.answer)
    if result.report.caveats:
        print("\nCaveats:")
        for caveat in result.report.caveats:
            print(f"- {caveat}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            'usage: python examples/hosted_cycle.py "describe the problem"'
        )
    asyncio.run(main(" ".join(sys.argv[1:])))
