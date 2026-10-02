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
├── institutional role
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

Several logical agents may share one loaded local model. Their independence is
created by isolated contexts, identities, instructions, memories, histories,
and protocol boundaries rather than by requiring separate model weights.

Changing a backend must not require changing the institutional kernel.

Changing an agent's instructions, memory policy, tools, role definition, or
backend creates a new agent version so its historical performance remains
meaningful.

## Institutional roles

MOSAIC uses three primary language-agent roles plus deterministic tools.

### Coordinator

The Coordinator is the professional interface between the human and the
research process.

Its responsibilities are:

- interpret user input
- separate observations from user interpretations
- normalize terminology and units where possible
- identify the investigation question
- identify missing or ambiguous information
- select relevant specialist Thinkers
- prepare neutral task packets
- track investigation stage
- receive completed findings
- produce a faithful final report for the user

The Coordinator does **not** conduct the scientific investigation.

It must not:

- originate scientific hypotheses during intake
- tell a Thinker which explanation is likely correct
- bias task packets with another Thinker's conclusions
- alter a Thinker's confidence or prediction
- suppress minority findings for presentation convenience
- resolve scientific disagreement by authority
- silently convert interpretation into observation
- present consensus as established fact

The Coordinator may summarize, organize, and explain findings, but its final
answer must remain traceable to the institutional record.

### Thinker

Thinkers perform the creative scientific work.

Each Thinker is a versioned specialist with an isolated first-pass context.
Examples include:

- mechanical / physical systems
- electrical / controls
- physics / first principles
- experimental science / statistics
- other domain specialists added as needed

A Thinker may:

- generate hypotheses
- derive mechanisms
- perform calculations
- use permitted tools
- identify assumptions
- produce falsifiable predictions
- suggest experiments
- revise its own proposal when challenged

Thinkers are encouraged to explore independently and may disagree.

A Thinker does not see another Thinker's first-pass conclusions unless the
protocol has explicitly entered cross-domain review.

### Examiner

The Examiner is the epistemic gate between a Thinker's raw proposal and a
finding that may enter institutional synthesis.

The Examiner receives one Thinker's proposal plus the relevant investigation
evidence and attempts to break the proposal.

Its responsibilities include:

- challenge unsupported claims
- identify hidden assumptions
- search for contradictions
- test internal consistency
- demand falsifiable predictions
- check whether confidence is justified
- distinguish correlation from mechanism
- identify confounding explanations
- call deterministic analytical tools
- ask the originating Thinker direct questions
- require revision when an answer is inadequate
- identify the evidence needed to resolve uncertainty

The Examiner is not a passive summarizer.

It directly conducts a private examination loop with the Thinker that produced
the targeted claim.

```text
Thinker proposal
      |
      v
   Examiner
      |
      +---- objection / question ----+
      |                              |
      v                              |
   Thinker response                  |
      |                              |
      +------------------------------+
      |
      v
Examiner disposition
```

The loop ends when the Examiner can issue a disposition, not merely when the
two models agree.

## Examiner dispositions

Every examined finding must end in one of the following states:

### ACCEPTED

The reasoning is sufficiently coherent and supported for the currently
available evidence.

Acceptance does not mean proven truth.

### ACCEPTED_WITH_RESERVATIONS

The finding is coherent enough to retain, but material assumptions,
uncertainties, or limitations remain.

The reservations must be recorded with the finding.

### UNRESOLVED

A critical question cannot be answered from the available evidence.

The Examiner should identify the missing observation or discriminating
experiment rather than forcing a conclusion.

### REJECTED

The proposal is internally inconsistent, contradicted by available evidence,
non-falsifiable in its current form, or unsupported after examination.

Rejected proposals remain in the audit history.

## Core investigation protocol

The default MOSAIC reasoning pipeline is:

```text
USER
 |
 v
COORDINATOR
 |
 |  extract observations
 |  define question
 |  select specialists
 |  create neutral task packets
 v
FROZEN INVESTIGATION SNAPSHOT
 |
 +-------------------+-------------------+
 |                   |                   |
 v                   v                   v
THINKER A          THINKER B          THINKER C
 |                   |                   |
 v                   v                   v
EXAMINER A         EXAMINER B         EXAMINER C
 |                   |                   |
 <---- private challenge / response loops ---->
 |                   |                   |
 v                   v                   v
EXAMINED FINDING A EXAMINED FINDING B EXAMINED FINDING C
          \            |            /
           \           |           /
            v          v          v
             CROSS-DOMAIN REVIEW
                     |
                     v
             HYPOTHESIS GRAPH
                     |
                     v
           EXPERIMENT SELECTION
                     |
                     v
                COORDINATOR
                     |
                     v
                    USER
```

The Coordinator should not receive raw Thinker output as an institutional
conclusion. It receives examined findings, dispositions, statistics,
reservations, unresolved questions, minority findings, and experiment
recommendations.

## Isolation rules

### First-pass isolation

All selected Thinkers receive the same frozen point-in-time investigation
snapshot for their first pass.

A Thinker's early completion must not change the context seen by another
Thinker in that same round.

### Examination isolation

The Examiner reviewing Thinker A should not initially see Thinker B's proposed
answer.

This prevents another specialist's conclusion from steering the Examiner's
challenge toward premature consensus.

Conceptually:

```text
Mechanical Thinker <--> Examiner session M

Electrical Thinker <--> Examiner session E

Physics Thinker    <--> Examiner session P
```

Only after each private examination has produced a disposition do findings
enter cross-domain review.

### Cross-domain review

Cross-domain review compares independently examined findings.

Its purpose is to detect:

- contradictory predictions
- overlapping explanations
- incompatible assumptions
- hidden shared dependencies
- opportunities for discriminating experiments
- cases where two apparently different hypotheses are equivalent

Cross-domain review does not erase minority findings.

## Agent boundary

Agents do not write directly to the event database.

The orchestration layer gives an agent a read-only investigation view. The
agent returns structured output. MOSAIC validates that output and records
accepted protocol events through the kernel.

Conceptually:

```text
ledger
  |
  v
investigation replay
  |
  v
read-only snapshot
  |
  v
role-specific agent
  |
  v
structured output
  |
  v
schema + protocol validation
  |
  v
append-only kernel events
```

This prevents provider-specific behavior from bypassing institutional rules.

## Proposed interfaces

The precise APIs may evolve, but role boundaries should remain explicit.

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

The next layer should add role-specific contracts such as:

```python
class Coordinator(Protocol):
    async def intake(...) -> InvestigationPlan: ...
    async def report(...) -> UserReport: ...


class Thinker(Protocol):
    async def investigate(...) -> ThinkerProposal: ...
    async def answer_examination(...) -> ThinkerResponse: ...


class Examiner(Protocol):
    async def examine(...) -> ExaminationResult: ...
```

Structured outputs are institutional records, not free-form authority.

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
HTTP interface.

### Shared local backend

A single loaded local model may initially power multiple institutional roles:

```text
                  shared local model
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
    Coordinator       Thinker        Examiner
     context           context         context
```

The roles remain logically independent because their prompts, contexts,
histories, permissions, and outputs are isolated.

This is the preferred initial deployment because it minimizes memory and
hardware requirements without weakening the institutional protocol.

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

The ledger should record backend identity and model/version provenance for every
agent run.

### Non-LLM agents and tools

Not every MOSAIC analytical component should be an LLM.

Useful deterministic components include:

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

The Examiner should make heavy use of deterministic analytical tools where they
can test a claim more reliably than another language-model inference.

### Human participants

A human researcher can also be represented as an attributed actor.

Human hypotheses and predictions can therefore enter the same graph without
receiving privileged treatment or being confused with raw observations.

## Dynamic specialist routing

MOSAIC does not need to invoke every Thinker for every investigation.

The Coordinator selects specialists based on the problem domain while keeping
the task packet neutral.

For example:

```text
Question: shaft vibration near 3200 rpm

Mechanical       invoke
Physics          invoke
Statistics       invoke
Electrical       optional
Economics        skip
```

This saves inference cost while preserving specialist independence.

Routing itself should later be evaluated as a measurable institutional skill.

## Credibility by role

Thinkers and Examiners should not share one generic credibility score.

### Thinker credibility

Possible measures include:

- prediction accuracy
- calibration
- useful hypothesis generation
- survival under later evidence
- useful minority predictions
- experiment quality

### Examiner credibility

Possible measures include:

- flaws correctly identified
- unsupported claims rejected
- useful unresolved questions surfaced
- discriminating experiments requested
- false rejection rate
- acceptance of claims later disproven
- calibration of its dispositions

### Coordinator quality

The Coordinator may later be evaluated on:

- correct observation extraction
- neutral routing
- preservation of uncertainty
- faithful representation of minority findings
- avoidance of interpretation leakage
- usefulness and accuracy of final reports

The Coordinator's quality score should not become scientific authority.

## Orchestration ownership

MOSAIC owns orchestration instead of delegating the institutional process to an
external multi-agent framework.

External agent frameworks may be used behind adapters when useful, but MOSAIC
remains responsible for:

- Coordinator discipline
- first-pass isolation
- examination isolation
- Thinker/Examiner dialogue
- agent identity/versioning
- hypothesis and prediction submission
- cross-domain review order
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
| investigations   hypothesis graph   experiments      |
| agent status     examinations       audit history    |
| predictions      minority reports   credibility      |
+---------------------------+--------------------------+
                            |
                     MOSAIC runtime
                            |
       +--------------------+--------------------+
       |                    |                    |
     kernel             orchestration          tools
       |                    |
    SQLite            role registry
                            |
             +--------------+--------------+
             |              |              |
        Coordinator       Thinkers       Examiners
             |              |              |
             +-------- shared backend -----+
                            |
                   local model server
                            |
                      optional remote
                         escalation
```

The GUI is a client of the same local runtime APIs used by CLI or automation.

## Current implementation rule

Do not implement autonomous free-running agents first.

The next agent milestones should be:

1. preserve the existing frozen `InvestigationSnapshot`
2. introduce explicit Coordinator, Thinker, and Examiner contracts
3. keep first-pass Thinker execution isolated
4. add structured examination questions and Thinker responses
5. require an Examiner disposition before a finding enters synthesis
6. persist examination transcripts and dispositions as audit events
7. add cross-domain comparison only after private examination is complete
8. then connect a real local model backend

Only after this protocol is reliable should MOSAIC add long-lived autonomous
scheduling or broader agent self-direction.
