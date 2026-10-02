# MOSAIC Agent Architecture

## Local-first requirement

MOSAIC is intended to become a complete local application.

The institutional kernel, event ledger, investigation state, orchestration,
agent definitions, memories, scoring, experiment logic, and user interface
must not require a hosted AI service to function.

Remote AI services are optional inference backends. They may supplement local
agents, but they are not the system of record and they do not own MOSAIC state.

A user should ultimately be able to disconnect the machine from the network
and still run a useful MOSAIC installation with local models and deterministic
agents.

## Agent is not model

MOSAIC deliberately separates an **agent** from the model that performs
inference for it.

An agent is a versioned research role:

```text
Agent
├── identity + version
├── specialty / domain scope
├── instructions / method
├── private memory policy
├── tool permissions
├── output schema
├── credibility history
└── inference backend
```

The inference backend is replaceable:

```text
                     MOSAIC Agent
                          |
              +-----------+------------+
              |           |            |
              v           v            v
          local LLM   remote LLM   deterministic
                                      program
```

Changing a backend must not require changing the institutional kernel.

Changing a specialist's instructions, memory policy, tools, or backend creates
a new agent version so its historical prediction record remains meaningful.

## Agent boundary

Agents do not write directly to the event database.

The orchestration layer gives an agent a read-only investigation view. The
agent returns a structured proposal. MOSAIC validates that proposal and, if it
is accepted by the current protocol, records the corresponding events.

Conceptually:

```text
ledger
  |
  v
investigation replay
  |
  v
read-only agent context
  |
  v
specialist agent
  |
  v
structured proposal
  |
  v
schema + policy validation
  |
  v
append-only kernel events
```

This prevents provider-specific behavior from bypassing institutional rules.

## Proposed interfaces

The precise Python API will be implemented after the epistemic primitives are
stable, but the boundary should resemble:

```python
class ModelBackend(Protocol):
    async def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        ...


class ResearchAgent(Protocol):
    identity: AgentIdentity

    async def analyze(
        self,
        context: InvestigationSnapshot,
    ) -> AgentProposal:
        ...
```

`AgentProposal` will be structured data, not free-form authority. It may
contain proposed hypotheses, predictions, criticisms, questions, or experiment
ideas.

## Backend classes

### Local backends

Local inference is the default deployment target.

MOSAIC should support local engines through adapters rather than embedding one
runtime deeply into the application.

Likely adapter families include:

- OpenAI-compatible local HTTP servers
- llama.cpp
- Ollama
- vLLM
- direct in-process model runtimes where useful
- deterministic Python/scientific modules

Where practical, the first generic adapter should target an OpenAI-compatible
HTTP interface. Multiple local serving projects expose such interfaces, which
lets one MOSAIC adapter cover several runtimes while still allowing
runtime-specific adapters later.

### Remote backends

Remote models are optional.

Provider-specific adapters can expose hosted models through the same
`ModelBackend` contract. Remote agents receive only the context explicitly
assembled for that invocation.

Possible policies include:

- local-only
- remote-allowed
- remote-preferred for selected domains
- local-first with remote escalation
- privacy-tagged observations that may never leave the machine

The ledger should record the backend identity and model/version associated with
every agent run so later calibration remains attributable.

### Non-LLM agents

Not every MOSAIC agent should be an LLM.

Useful agents may be deterministic or numerical:

- dimensional-analysis checker
- unit-consistency checker
- statistical test selector
- uncertainty propagator
- curve fitter
- simulator
- constraint solver
- symbolic algebra system
- rules engine
- anomaly detector

These can participate in the same proposal and credibility framework as
language-model agents.

### Human participants

A human researcher can also be represented as an attributed actor.

Human hypotheses and predictions can therefore enter the same graph without
receiving privileged treatment or being confused with raw observations.

## Initial specialist population

The first useful engineering deployment is expected to use approximately five
specialists:

1. mechanical / physical systems
2. electrical / controls
3. physics / first principles
4. experimental science / statistics
5. adversarial generalist

They should perform the first-pass interpretation independently.

The same local model may initially power several specialists, but each
specialist must have a distinct identity, instructions, private memory, and
prediction history.

Later, diversity can be increased by assigning different local or remote models
to different specialists.

## Orchestration

MOSAIC should own orchestration instead of delegating the institutional process
to an external multi-agent framework.

External agent frameworks may be used behind adapters when they provide useful
execution capabilities, but MOSAIC must remain responsible for:

- isolation of first-pass reasoning
- agent identity/versioning
- hypothesis and prediction submission
- adversarial review order
- experiment selection protocol
- credibility calculation
- event recording
- replay
- methodological lineage

This keeps the research protocol reproducible even when model providers or
agent libraries change.

## Local application direction

A mature local MOSAIC installation can be organized as:

```text
+------------------------------------------------------+
|                    MOSAIC Desktop                    |
|                                                      |
|  investigations   hypothesis graph   experiments     |
|  agent status     prediction scores  audit history   |
+---------------------------+--------------------------+
                            |
                     MOSAIC runtime
                            |
          +-----------------+-----------------+
          |                 |                 |
        kernel         orchestration       tools
          |                 |
       SQLite         agent registry
                            |
              +-------------+-------------+
              |             |             |
          local LLM     local LLM     remote LLM
              |             |          (optional)
        llama.cpp /       vLLM /
          Ollama          others
```

The GUI is a client of the same local runtime APIs used by CLI or automation.
The desktop interface should therefore remain replaceable without coupling the
research state to a particular UI toolkit.

## First implementation rule

Do not implement autonomous free-running agents first.

The first agent milestone should be:

1. freeze an `InvestigationSnapshot`
2. send the same snapshot independently to several agents
3. require schema-valid `AgentProposal` responses
4. record proposals with agent/model provenance
5. expose the proposals for adversarial review

Only after that loop is reliable should MOSAIC add tool execution, long-lived
agent memory, or autonomous scheduling.
