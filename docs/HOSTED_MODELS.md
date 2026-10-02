# Hosted model quick start

MOSAIC can now run its institutional roles against an OpenAI-compatible
Chat Completions endpoint.

The backend is intentionally provider-neutral. The same runtime can point at a
hosted service now and a compatible local server later.

## Configuration

Install the project in editable mode:

\`\`\`bash
python -m pip install -e ".[dev]"
\`\`\`

Set a model name:

\`\`\`bash
export MOSAIC_MODEL="<provider-model-name>"
\`\`\`

For the OpenAI API, the default base URL is already
\`https://api.openai.com/v1\`. Supply the key through either:

\`\`\`bash
export OPENAI_API_KEY="..."
\`\`\`

or:

\`\`\`bash
export MOSAIC_API_KEY="..."
\`\`\`

For another OpenAI-compatible server:

\`\`\`bash
export MOSAIC_BASE_URL="https://provider.example/v1"
export MOSAIC_MODEL="<model>"
export MOSAIC_API_KEY="..."
\`\`\`

The adapter does not write credentials into MOSAIC's ledger.

## Run

\`\`\`bash
python examples/hosted_cycle.py \
  "At light throttle near 1.2 ms injector pulse width it goes lean briefly."
\`\`\`

By default the example writes the append-only event ledger to \`mosaic.db\`.
Override it with:

\`\`\`bash
export MOSAIC_DB="/path/to/investigation.db"
\`\`\`

## What happens

One invocation runs:

\`\`\`text
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
\`\`\`

All language roles may use the same configured model. They remain separate
institutional agents because their identities, prompts, contexts, duties, and
validation boundaries are distinct.

## Local migration

To move the same runtime to a local compatible server, change
\`MOSAIC_BASE_URL\` and \`MOSAIC_MODEL\`. If the local server does not support
strict JSON-schema response formatting, instantiate
\`OpenAICompatibleBackend(..., structured_outputs=False)\` to use JSON-object
mode while retaining MOSAIC-side validation.
