"""Run one MOSAIC investigation against an OpenAI-compatible model server."""

from __future__ import annotations

import asyncio
import os
import sys
import time

from mosaic import AgentIdentity, Investigation, SQLiteEventStore
from mosaic.agents import (
    ModelCoordinator,
    ModelCrossDomainReviewer,
    ModelExaminer,
    ModelThinker,
    ProgressEvent,
    ResearchCycle,
    TrackedBackend,
    UsageTracker,
    openai_standard_pricing,
)
from mosaic.backends import OpenAICompatibleBackend


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} must be set")
    return value


class ConsoleProgress:
    """Persistent stage log plus a heartbeat while the current call is running."""

    _frames = "|/-\\"

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.stage = "startup"
        self.message = "Initializing MOSAIC."
        self.running = True
        self._width = 0

    def _clear_pulse(self) -> None:
        if self._width:
            sys.stdout.write("\r" + (" " * self._width) + "\r")
            sys.stdout.flush()
            self._width = 0

    def on_event(self, event: ProgressEvent) -> None:
        self._clear_pulse()
        elapsed = time.monotonic() - self.started
        print(
            f"[{elapsed:6.1f}s] [{event.stage.upper()}] {event.message}",
            flush=True,
        )
        self.stage = event.stage
        self.message = event.message

    async def pulse(self) -> None:
        index = 0
        try:
            while self.running:
                elapsed = time.monotonic() - self.started
                frame = self._frames[index % len(self._frames)]
                line = (
                    f"[MOSAIC {frame}] {self.stage}: {self.message} "
                    f"| elapsed {elapsed:6.1f}s"
                )
                padding = max(self._width - len(line), 0)
                sys.stdout.write("\r" + line + (" " * padding))
                sys.stdout.flush()
                self._width = len(line)
                index += 1
                await asyncio.sleep(0.25)
        finally:
            self._clear_pulse()

    def stop(self) -> None:
        self.running = False


def print_section(title: str, items: tuple[str, ...]) -> None:
    if not items:
        return
    print(f"\n{title}")
    for item in items:
        print(f"- {item}")


def print_report(result) -> None:
    report = result.report
    print("\nCURRENT FINDING")
    print(report.answer)
    print_section("OBSERVED", report.observations)
    print_section("ESTABLISHED", report.established)
    print_section("SURVIVING HYPOTHESES", report.surviving_hypotheses)

    if report.key_test:
        print("\nKEY DISCRIMINATING TEST")
        print(report.key_test)

    print_section("EXAMINER STATUS", report.examiner_status)
    print_section("UNRESOLVED QUESTIONS", report.unresolved_questions)
    print_section("CAVEATS", report.caveats)


def print_usage(tracker: UsageTracker) -> None:
    by_tag = tracker.by_tag()
    total = tracker.total()
    if not by_tag:
        return

    print("\nMODEL USAGE")
    print(
        f"{'ROLE / CALL':<42} {'IN':>9} {'OUT':>9} "
        f"{'CACHED':>9} {'EST. COST':>12}"
    )
    print("-" * 84)

    for tag, usage in by_tag.items():
        cost = (
            "n/a"
            if usage.estimated_cost_usd is None
            else f"USD {usage.estimated_cost_usd:.5f}"
        )
        print(
            f"{tag:<42} "
            f"{usage.input_tokens:>9,} "
            f"{usage.output_tokens:>9,} "
            f"{usage.cached_input_tokens:>9,} "
            f"{cost:>12}"
        )

    total_cost = (
        "n/a"
        if total.estimated_cost_usd is None
        else f"USD {total.estimated_cost_usd:.5f}"
    )
    print("-" * 84)
    print(
        f"{'TOTAL':<42} "
        f"{total.input_tokens:>9,} "
        f"{total.output_tokens:>9,} "
        f"{total.cached_input_tokens:>9,} "
        f"{total_cost:>12}"
    )
    if total.estimated_cost_usd is None:
        print(
            "Cost unavailable for this model/provider; token counts are still exact "
            "when the backend returns usage."
        )


async def main(user_input: str) -> None:
    model = require_env("MOSAIC_MODEL")
    base_url = os.getenv(
        "MOSAIC_BASE_URL",
        "https://api.openai.com/v1",
    )
    api_key = os.getenv("MOSAIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    database = os.getenv("MOSAIC_DB", "mosaic.db")

    raw_backend = OpenAICompatibleBackend(
        backend_id="primary",
        base_url=base_url,
        api_key=api_key,
        api_key_env=None,
        structured_outputs=True,
    )

    price = openai_standard_pricing(model)
    tracker = UsageTracker(
        pricing={} if price is None else {model: price},
    )
    backend = TrackedBackend(raw_backend, tracker)

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

    progress = ConsoleProgress()
    cycle = ResearchCycle(
        coordinator=coordinator,
        thinkers=thinkers,
        examiner=examiner,
        reviewer=reviewer,
        progress=progress.on_event,
    )

    pulse_task = asyncio.create_task(progress.pulse())
    try:
        with SQLiteEventStore(database) as store:
            investigation = Investigation(store, "interactive")
            result = await cycle.run(
                investigation=investigation,
                user_input=user_input,
            )
    finally:
        progress.stop()
        await pulse_task

    print_report(result)
    print_usage(tracker)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            'usage: python examples/hosted_cycle.py "describe the problem"'
        )
    asyncio.run(main(" ".join(sys.argv[1:])))
