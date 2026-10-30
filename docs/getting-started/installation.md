# Installation

domyn-swarm needs Python 3.11 or newer.

## From PyPI

```bash
pip install domyn-swarm
```

With [uv](https://docs.astral.sh/uv/):

```bash
# into the current environment
uv pip install domyn-swarm

# as a project dependency; add --extra lepton for the Lepton backend
uv add domyn-swarm

# as a standalone tool
uv tool install --python 3.12 domyn-swarm
```

## Optional extras

The base install serves models and runs jobs with pandas. Everything else is an
extra, so a minimal install stays small:

| Extra | Adds | Install |
| --- | --- | --- |
| `lepton` | NVIDIA DGX Cloud Lepton backend, via the Lepton SDK | `pip install 'domyn-swarm[lepton]'` |
| `polars` | Polars data backend — `scan_parquet`, streaming reads, directory outputs | `pip install 'domyn-swarm[polars]'` |
| `ray` | Ray data backend for distributed execution | `pip install 'domyn-swarm[ray]'` |
| `all` | All of the above | `pip install 'domyn-swarm[all]'` |

See [Choosing a data backend](../guides/data-backends.md) for when `polars` or
`ray` are worth the extra dependency.

## From source

Install from the repository to use a change that is not released yet. To pin a
revision, replace `main` with a branch name, a tag, or a commit hash:

```bash
pip install git+https://github.com/DomynAI/domyn-swarm.git@main
```

With uv:

```bash
uv tool install --from git+https://github.com/DomynAI/domyn-swarm.git@main \
  --python 3.12 domyn-swarm
```

## Verify

```bash
domyn-swarm version
```

The CLI is also installed as `ds` and `dswarm`, which are the same entry point.

## Lepton setup

Running on DGX Cloud Lepton needs the extra plus credentials:

```bash
pip install 'domyn-swarm[lepton]'
lep login
```

`lep login` initialises the credentials domyn-swarm reads when it creates
endpoints and jobs. See [Running on Lepton](../guides/lepton.md).

## Next steps

Launch your first swarm in [Quickstart](quickstart.md).
