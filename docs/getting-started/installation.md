# Installation

domyn-swarm needs Python 3.11 or newer.

## From PyPI

:::{note}
The package is not published on PyPI yet, but it will be soon. Until then, use
[from source](#from-source).
:::

```bash
pip install domyn-swarm
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

Pin a release rather than tracking the default branch:

```bash
RELEASE=v0.34.0
pip install git+ssh://git@github.com/DomynAI/domyn-swarm.git@$RELEASE
```

With [uv](https://docs.astral.sh/uv/):

```bash
RELEASE=v0.34.0

# into the current environment
uv pip install git+ssh://git@github.com/DomynAI/domyn-swarm.git@$RELEASE

# as a project dependency; add --extra lepton for the Lepton backend
uv add git+ssh://git@github.com/DomynAI/domyn-swarm.git@$RELEASE

# as a standalone tool
uv tool install --from git+ssh://git@github.com/DomynAI/domyn-swarm.git@$RELEASE \
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
