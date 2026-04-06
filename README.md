# eval-framework-py

Project-agnostic evaluation framework for running dataset-based evals with deterministic and OpenAI-first judge strategy defaults.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## CLI

```bash
python -m eval_framework.cli run --dataset data/evals/sample_offer_parser_redacted.json --task oracle_expected
python -m eval_framework.cli compare --left runs/run_a.json --right runs/run_b.json
python -m eval_framework.cli dataset add data/evals/sample_offer_parser_redacted.json --example '{"example_id":"ex-2","input":{},"expected":{}}'
```
