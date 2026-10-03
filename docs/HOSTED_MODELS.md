# Hosted model quick start

MOSAIC can run its institutional roles against an OpenAI-compatible Chat
Completions endpoint.

The backend is provider-neutral. The same runtime can point at a hosted service
now and a compatible local server later.

## Configuration

Install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Set a model name:

```bash
export MOSAIC_MODEL="<provider-model-name>"
```

For the OpenAI API, the default base URL is `https://api.openai.com/v1`.
Supply the key through either:

```bash
export OPENAI_API_KEY="..."
```

or:

```bash
export MOSAIC_API_KEY="..."
```

For another OpenAI-compatible server:

```bash
export MOSAIC_BASE_URL="https://provider.example/v1"
export MOSAIC_MODEL="<model>"
export MOSAIC_API_KEY="..."
```

The adapter does not write credentials into MOSAIC's ledger.

## Run a fresh investigation

```bash
python examples/hosted_cycle.py \
  "At light throttle near 1.2 ms injector pulse width it goes lean briefly."
```

Every invocation creates a fresh investigation stream by default. The CLI prints
the generated ID, for example:

```text
[INVESTIGATION] INV-20261003-051412-a8f3c219 (new)
```

The append-only event ledger still defaults to `mosaic.db`, so multiple
independent investigations can live in the same database without contaminating
one another.

Override the database path with:

```bash
export MOSAIC_DB="/path/to/investigation.db"
```

## Continue an existing investigation

Continuation is explicit:

```bash
python examples/hosted_cycle.py \
  --continue INV-20261003-051412-a8f3c219 \
  "I repeated the test with AE disabled and the lean event remained."
```

If the requested investigation ID does not exist in the configured database,
the CLI exits instead of silently starting a new stream.

## Interactive clarification review

By default, an interactive terminal run may pause after Coordinator intake and
before the Thinkers if the Coordinator identifies up to three high-value facts
that the user may already know. The heartbeat pauses while MOSAIC is waiting
for input.

Answer each question at the prompt, or press Enter to skip an item. Answers are
stored in the same investigation and Coordinator intake is refreshed once
before specialist analysis begins.

For unattended or scripted runs, disable the prompt with:

```bash
python examples/hosted_cycle.py --no-review "describe the problem"
```

Non-interactive stdin also skips the review prompt automatically.

## What happens

One invocation runs:

```text
user input
   |
Coordinator intake
   |
reported observations -> ledger
   |
frozen investigation snapshot
   |
selected Thinkers
   |
private Thinker <-> Examiner loops
   |
cross-domain synthesis
   |
controlled graph promotion
   |
Coordinator report
```

All language roles may use the same configured model. They remain separate
institutional agents because their identities, prompts, contexts, duties, and
validation boundaries are distinct.

Role-specific model contexts are intentionally compact. Thinkers and Examiners
receive task-relevant observations plus prior institutional claims; synthesis
receives examined findings without audit-heavy fields; the final Coordinator
receives concise findings and synthesis rather than the full promoted graph.

## Local migration

To move the same runtime to a local compatible server, change
`MOSAIC_BASE_URL` and `MOSAIC_MODEL`. If the local server does not support
strict JSON-schema response formatting, instantiate
`OpenAICompatibleBackend(..., structured_outputs=False)` to use JSON-object
mode while retaining MOSAIC-side validation.
