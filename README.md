# MOSAIC

**Multi-agent Observational Synthesis, Adversarial Inquiry, and Calibration**

> **Evolve the process of finding out, not the answers.**

MOSAIC is an experimental architecture for persistent, adversarial, multi-agent scientific reasoning.

Rather than asking one model to produce a conclusion, MOSAIC treats investigation as a structured process. Independent specialist agents receive shared observations, develop interpretations separately, make falsifiable predictions, challenge competing hypotheses, propose discriminating experiments, and earn domain-specific credibility from how well their predictions survive contact with reality.

The long-term objective is not merely a system that answers questions more accurately. It is a system that can improve the **method by which it discovers answers** while preserving a complete, auditable history of observations, hypotheses, predictions, experiments, failures, minority reports, and methodological changes.

## Core idea

MOSAIC separates five things that are often collapsed together:

1. **Observation** — what was actually measured, recorded, or supplied.
2. **Interpretation** — what an agent believes those observations may mean.
3. **Hypothesis** — an explicit explanatory model.
4. **Prediction** — what should be observed if that hypothesis is correct.
5. **Experiment** — a deliberate test chosen to distinguish competing predictions.

An observation such as:

```text
Injector pulse width: 1.31 ms
MAP:                  42 kPa
Engine speed:         875 rpm
O2 response delay:    340 ms
```

is stored as observation.

A statement such as:

```text
The transient was caused by manifold wall-film behavior.
```

is not.

That is a hypothesis, and MOSAIC requires it to compete with alternatives.

## System heartbeat

```text
                    OBSERVATIONS
                         |
                         v
              independent interpretation
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
        specialist   specialist   specialist
          agent         agent         agent
             |           |           |
             +------ hypotheses ------+
                         |
                         v
                 attach predictions
                         |
                         v
                 adversarial review
                         |
                         v
                  hypothesis graph
                         |
                         v
              identify key uncertainty
                         |
                         v
             design discriminating test
                         |
                         v
               execute / request test
                         |
                         v
                 record observation
                         |
                         v
                score predictions
                         |
                         v
              update domain credibility
                         |
                         +-----> repeat
```

The system is intended to become better at asking the next useful question, not simply better at sounding certain.

## Architecture

### 1. Institutional kernel

The kernel is deliberately small and deliberately unintelligent.

Its responsibilities are infrastructure:

- immutable event logging
- agent and version identity
- permissions
- experiment execution records
- timestamps
- audit trails
- reproducibility
- version history
- rollback references
- scheduling and orchestration

The kernel does **not** decide which scientific interpretation is correct.

It should also be the hardest part of MOSAIC to modify. Higher-level components may evolve; the machinery that preserves history and accountability should not casually rewrite itself.

### 2. Observation ledger

Observations are stored independently from conclusions.

Every observation should retain provenance such as:

- source
- timestamp
- units
- uncertainty
- acquisition method
- experiment ID
- transformation history
- raw-data reference

Interpretation belongs elsewhere.

This allows MOSAIC to revisit old evidence when a new hypothesis appears without inheriting the assumptions of the original analysis.

### 3. Independent specialist agents

MOSAIC begins with a small population of specialists rather than a large swarm.

An initial engineering-oriented set may include:

- mechanical / physical systems
- electrical / controls
- physics / first-principles analysis
- experimental science / statistics
- adversarial generalist

Each agent receives the same observation packet but performs its first interpretation **independently**.

Agents should maintain separate domain histories and private working memories so that one agent's interpretation does not immediately contaminate the reasoning of the others.

Only after independent hypotheses are recorded does cross-agent deliberation begin.

### 4. Hypothesis graph

MOSAIC stores a persistent graph rather than a single conclusion.

A simplified structure:

```text
Observation O17
     |
     +-- H12: injector nonlinearity
     |      +-- predicts P31
     |      +-- predicts P32
     |      +-- contradicted by O24
     |
     +-- H13: wall-film transient
     |      +-- predicts P35
     |      +-- supported by O29
     |
     +-- H14: O2 transport delay
            +-- predicts P41
            +-- unresolved
```

Graph edges may represent relationships such as:

- supports
- contradicts
- predicts
- derived from
- alternative to
- depends on
- supersedes
- unresolved by
- tested by

A hypothesis that produces no discriminating prediction has limited epistemic value.

### 5. Adversarial review

Consensus is not treated as truth.

After independent reasoning, agents may be assigned temporary roles such as:

- **Proponent** — defend a hypothesis.
- **Critic** — identify hidden assumptions and failure modes.
- **Alternative theorist** — construct a different explanation for the same observations.
- **Cross-domain reviewer** — search for effects outside the originating specialty.
- **Arbiter** — identify unresolved disagreements and the evidence needed to resolve them.

The purpose of review is not to force agreement. It is to expose uncertainty.

### 6. Minority reports

Losing hypotheses are not deleted.

A low-support hypothesis may later explain a new observation better than the former consensus.

MOSAIC therefore preserves:

- rejected hypotheses
- minority hypotheses
- dissenting predictions
- failed experiments
- reasoning lineage
- confidence at the time a prediction was made

This creates institutional memory of both success and error.

### 7. Prediction registry

Predictions are committed **before** the relevant result is known.

A prediction record should include:

- hypothesis ID
- agent/version ID
- domain
- predicted observation
- confidence
- tolerances or expected range
- conditions under which the prediction applies
- timestamp
- scoring rule

This prevents post-hoc reinterpretation from being mistaken for predictive success.

### 8. Experiment designer

When several hypotheses explain the available observations, MOSAIC should search for the experiment whose possible outcomes separate them most efficiently.

Conceptually:

```text
experiment value
    =
expected information gain
× discrimination power
-------------------------
cost + time + risk
```

The exact utility model can evolve later.

The key principle is that the experiment designer asks:

> What is the cheapest, safest, most discriminating observation we can obtain next?

This turns the system from a passive analyst into an active research process.

### 9. Dynamic credibility

Agents do not receive permanent global authority.

Credibility is domain-specific and earned through measurable performance.

For example:

```text
Agent Mechanical-01

rotating machinery       0.91
fluid systems            0.74
combustion               0.63
electronics              0.27
statistical calibration  0.81
```

Potential scoring inputs include:

- prediction accuracy
- calibration error
- false-confidence rate
- useful minority predictions
- experiment quality
- hypothesis survival
- ability to identify contradictions
- performance on held-out problems

Influence should follow demonstrated calibration, not rhetorical confidence.

## Methodological evolution

Self-modification occurs one level above ordinary reasoning.

A separate evolutionary manager may eventually propose changes to:

- agent prompts
- reasoning protocols
- memory retrieval strategies
- specialist composition
- review procedures
- experiment-selection functions
- scoring systems
- hypothesis-generation methods

It may **not** silently rewrite the institutional kernel or historical record.

A proposed method change becomes a new version and must prove itself against:

- held-out problems
- replayed historical investigations
- controlled benchmarks
- new real-world experiments

Example:

```text
Mechanical-02:v7
      |
      +-- mutation A --> v8a
      +-- mutation B --> v8b
      +-- mutation C --> v8c
                         |
                         v
                   benchmark suite
                         |
                         v
                    v8b promoted
```

Old versions and their histories remain reproducible.

The scientific method is therefore applied not only to external questions, but eventually to MOSAIC's own research procedures.

## Initial MVP

The first version should stay intentionally small.

### Phase 1

- Python core
- SQLite event store
- immutable event schema
- observation model
- hypothesis model
- prediction model
- experiment model
- agent/version identities
- append-only audit trail

### Phase 2

- five specialist agents
- isolated first-pass reasoning
- structured hypothesis submission
- structured prediction submission
- hypothesis graph
- adversarial cross-examination
- minority-report preservation

### Phase 3

- prediction scoring
- confidence calibration
- domain-specific credibility
- replay of previous investigations
- comparison of agent/version performance

### Phase 4

- experiment proposal generation
- experiment discrimination scoring
- cost / risk / time metadata
- human approval and execution interface
- result ingestion

### Phase 5

- methodological mutation proposals
- sandbox evaluation
- held-out benchmark suite
- lineage tracking
- promotion / rejection of candidate methods

The evolutionary layer should not be implemented until the earlier layers produce enough history to evaluate changes meaningfully.

## Proposed repository structure

```text
mosaic/
|
+-- kernel/
|   +-- events.py
|   +-- identity.py
|   +-- permissions.py
|   +-- scheduler.py
|   +-- audit.py
|
+-- knowledge/
|   +-- observations.py
|   +-- hypotheses.py
|   +-- predictions.py
|   +-- experiments.py
|   +-- hypothesis_graph.py
|
+-- agents/
|   +-- base_agent.py
|   +-- mechanical.py
|   +-- electrical.py
|   +-- physics.py
|   +-- statistics.py
|   +-- critic.py
|
+-- deliberation/
|   +-- independent_pass.py
|   +-- cross_exam.py
|   +-- minority_reports.py
|   +-- synthesis.py
|
+-- experiments/
|   +-- designer.py
|   +-- scorer.py
|   +-- executor.py
|
+-- reputation/
|   +-- calibration.py
|   +-- credibility.py
|   +-- domains.py
|
+-- evolution/
|   +-- proposal.py
|   +-- sandbox.py
|   +-- benchmark.py
|   +-- promotion.py
|
+-- models/
|   +-- providers.py
|
+-- storage/
|   +-- research.db
|
+-- tests/
|
+-- main.py
```

This structure is provisional. The event model and information boundaries matter more than preserving a particular directory layout.

## Example investigation

Eventually, an investigation should look conceptually like:

```python
institution.investigate(
    observations=[
        Observation("RPM", 875, "rpm"),
        Observation("MAP", 42, "kPa"),
        Observation("injector_pw", 1.31, "ms"),
    ],
    question="What mechanisms could explain the observed transient?"
)
```

The useful output is not merely prose.

It is a structured research state containing competing hypotheses, predictions, evidence relationships, criticisms, uncertainties, credibility-weighted assessments, and proposed experiments.

Conceptually:

```json
{
  "hypotheses": [
    {
      "id": "H17",
      "claim": "Injector low-pulse-width nonlinearity",
      "proposed_by": "mechanical_01:v1",
      "confidence": 0.58,
      "predictions": ["P31", "P32"],
      "supporting_observations": ["O1", "O4"],
      "contradictions": ["O7"],
      "status": "active"
    }
  ]
}
```

## Initial proving ground

Early development will use engineering diagnosis problems where controlled observations and experiments are practical.

Fuel-injection and engine-control diagnostics are particularly useful because they combine:

- mechanical systems
- electrical systems
- control theory
- combustion physics
- sensor transport delay
- nonlinear actuators
- transient behavior
- noisy measurements
- competing causal explanations

This provides a real environment in which MOSAIC can be judged by whether it learns to propose **better discriminating tests**, rather than by whether its explanations merely sound convincing.

## Design principles

**Observations are not interpretations.**

**Predictions must be committed before outcomes are known.**

**Consensus is a policy, not truth.**

**Minority reports survive.**

**Confidence does not equal credibility.**

**Credibility is earned by calibrated prediction.**

**History is append-only and auditable.**

**Method changes must compete against the methods they replace.**

**The kernel preserves the institution; it does not conduct the science.**

**MOSAIC evolves the process of finding out, not the answers.**

## Status

MOSAIC is currently in the **architecture and initial implementation phase**.

The immediate development target is the smallest complete epistemic loop:

```text
observation
    -> independent hypotheses
    -> predictions
    -> adversarial criticism
    -> experiment selection
    -> new observation
    -> prediction scoring
    -> credibility update
    -> repeat
```

If that loop becomes measurably better at selecting useful questions and experiments over repeated investigations, MOSAIC will have demonstrated its central premise.
