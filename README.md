<div align="center">

<img src="static/domyn-swarm-logo-primary.svg#gh-light-mode-only" alt="Domyn Swarm" height="100">
<img src="static/domyn-swarm-logo-white.svg#gh-dark-mode-only" alt="Domyn Swarm" height="100">

**Deploy LLMs and run resumable batch inference on Slurm and NVIDIA DGX Cloud Lepton.**

domyn-swarm is a CLI and Python library that combines model deployment and batch processing in one
workflow. vLLM handles inference. domyn-swarm manages replicas, load balancing, and job execution.

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://domynswarm.domyn.com/)
[![CI](https://github.com/DomynAI/domyn-swarm/actions/workflows/ci.yaml/badge.svg)](https://github.com/DomynAI/domyn-swarm/actions/workflows/ci.yaml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/DomynAI/domyn-swarm/badges/coverage.json)](https://github.com/DomynAI/domyn-swarm/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-brightgreen)](https://github.com/DomynAI/domyn-swarm/blob/main/pyproject.toml)
[![License - Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://github.com/microsoft/pyright)

<a href="#quick-start">Quick start</a> •
<a href="#write-your-own-job">Custom jobs</a> •
<a href="#backends">Backends</a> •
<a href="#documentation">Documentation</a> •
<a href="CONTRIBUTING.md">Contributing</a>

</div>

[![Domyn Swarm CLI walkthrough: configure, launch, inspect, submit a batch job, monitor, and shut down a swarm](static/domyn-swarm-cli-demo.gif)](static/domyn-swarm-cli-demo.mp4)

<p align="center"><sub>CLI walkthrough — click the demo to open the MP4.</sub></p>

---

<table>
<tr>
<td width="50%" valign="top">

### ⚡ Serve

- **YAML configuration** for model and replica settings
- **OpenAI-compatible endpoints** backed by vLLM
- **Multiple replicas** behind a single load-balanced URL
- **Lifecycle commands** to launch, inspect, and stop named swarms

</td>
<td width="50%" valign="top">

### 🚀 Run

- **Batch inference** over Parquet datasets
- **Checkpointing and resume** for interrupted jobs
- **Retries with backoff** and **bounded concurrency**
- **Input sharding** for parallel job execution

</td>
</tr>
</table>

On Slurm, optional [Prometheus metrics and dashboards](https://domynswarm.domyn.com/latest/guides/metrics.html)
show inference throughput, queue depth, and GPU utilization.

---

## Quick start

> [!NOTE]
> domyn-swarm is not yet published on PyPI. Install it from the repository.
> `pip install domyn-swarm` will work after the first published release.

```bash
uv tool install --from git+https://github.com/DomynAI/domyn-swarm.git@v0.34.0 --python 3.12 domyn-swarm
```

<details>
<summary>Extras and other install methods</summary>

```bash
# Optional extras
pip install 'domyn-swarm[lepton]'   # NVIDIA DGX Cloud Lepton backend
pip install 'domyn-swarm[polars]'   # Polars data backend
pip install 'domyn-swarm[ray]'      # Ray Data backend
pip install 'domyn-swarm[all]'      # everything
```

`uv add`, editable installs and the Lepton `lep login` step are covered in
[Installation](https://domynswarm.domyn.com/latest/getting-started/installation.html).

</details>

**1. Describe the swarm**

The [`examples/configs/`](examples/configs/) directory holds ready-made configurations. This one
serves NVIDIA Nemotron-3-Super-120B-A12B across two replicas of four GPUs each, following the
[vLLM recipe](https://recipes.vllm.ai/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
([`nemotron_3_super.yaml`](examples/configs/nemotron_3_super.yaml)). Set the image paths,
partition, account, and QoS for your cluster:

```yaml
# NVIDIA Nemotron-3-Super-120B-A12B (BF16) — a ~120B latent-MoE with ~12B active
# parameters. Serving flags follow https://recipes.vllm.ai/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
# The recipe's `--served-model-name` is deliberately omitted: jobs address the
# endpoint by the `model` value below, so renaming the served model breaks them.
model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
name: nemotron-3-super-120b
revision: main
gpus_per_replica: 4       # tensor-parallel size 4, one node on most clusters
replicas: 2               # two independent replicas behind the load balancer
image: "/path/to/vllm.sif"   # needs vLLM >= 0.17.1
args: >-
  --tensor-parallel-size 4
  --kv-cache-dtype fp8
  --max-model-len 262144
  --trust-remote-code
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
  --reasoning-parser nemotron_v3
env:
  HF_HOME: /path/to/shared_hf_cache
wait_endpoint_s: 3600
backend:
  type: slurm
  partition: partition_name
  account: account_name
  qos: qos_name
  endpoint:
    port: 9001
    cpus_per_task: 4
    wall_time: "36:00:00"
    nginx_image: "/path/to/nginx.sif"
```

### 2. Launch the swarm

`up` submits the replica and load-balancer jobs and returns once the endpoint answers a health check.
Every swarm name combines the configured name with a short suffix. `up` writes
this name to standard output so that a script can capture it:

```console
$ SWARM=$(domyn-swarm up -c examples/configs/nemotron_3_super.yaml)
[08/31/26 11:56:44] INFO     Creating deployment nemotron-3-super-120b on slurm...
                    INFO     Submitted replicas job 15427318 with command: sbatch --parsable …
                    INFO     Submitted load balancer job 15427319 with command: sbatch --parsable …
                    INFO     LB healthy → http://lrdn0431:9001/v1

$ echo $SWARM
nemotron-3-super-120b-01m1bq26m7
```

### 3. Run a batch job

```console
$ domyn-swarm job submit \
    --name $SWARM \
    --input examples/data/chat_completion.parquet \
    --output results.parquet
[MainThread] Checkpoint flushed 16 rows, new total: 16
[MainThread] Checkpoint flushed 16 rows, new total: 32
[MainThread] Checkpoint flushed 16 rows, new total: 48
[MainThread] Checkpoint flushed 16 rows, new total: 64
[MainThread] Checkpoint flushed 1 rows, new total: 65
[MainThread] Processing all items in worker: 100%|██████████| 65/65 [01:12<00:00,  1.11s/sample]
```

This is the built-in chat-completion job reading the `messages` column. The example dataset has 65
rows. By default, the job writes a checkpoint after every 16 rows. An interrupted
run continues from the last checkpoint. A higher `replicas` value distributes
requests across more replicas.

### 4. Inspect and stop the swarm

```console
$ domyn-swarm status $SWARM      # live panel: replica states, endpoint, recorded jobs

$ domyn-swarm down -y $SWARM
✅ Swarm nemotron-3-super-120b-01m1bq26m7 shutdown request sent.
```

The [Quickstart](https://domynswarm.domyn.com/latest/getting-started/quickstart.html) covers the same
run step by step.

---

## Write your own job

Subclass `SwarmJob` and implement `transform_items` to define how inputs are processed.
The framework handles concurrency, retries, and checkpointing.

```python
from typing import Any

from domyn_swarm.jobs import SwarmJob


class SentimentJob(SwarmJob):
    """Classify each prompt as positive, negative, or neutral."""

    async def transform_items(self, items: list[Any]) -> list[str]:
        results = []
        for prompt in items:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Reply only: positive, negative, or neutral."},
                    {"role": "user", "content": prompt},
                ],
            )
            results.append(resp.choices[0].message.content or "")
        return results
```

This example uses the swarm's configured client and writes labels to the default `results` column.
See [Your first custom job](https://domynswarm.domyn.com/latest/getting-started/first-custom-job.html)
for input and output requirements, import setup, and CLI submission.

Built-in jobs cover chat completion, multi-turn conversations, perplexity, and translation.
See the [Jobs API reference](https://domynswarm.domyn.com/latest/reference/api/jobs.html).

---

## Backends

The same core commands are available on both platforms: `up`, `job submit`,
`status`, and `down`.

- Slurm is included in the base installation. See [Slurm setup](https://domynswarm.domyn.com/latest/guides/slurm.html).
- NVIDIA DGX Cloud Lepton requires the `lepton` extra and workspace credentials. See [Lepton setup](https://domynswarm.domyn.com/latest/guides/lepton.html).

Job input and output support pandas (default), polars, and Ray Data.
See [Choosing a data backend](https://domynswarm.domyn.com/latest/guides/data-backends.html).

To add support for another platform, see [Implementing a backend](https://domynswarm.domyn.com/latest/guides/implementing-a-backend.html).

---

## Documentation

The full documentation is published at **[domynswarm.domyn.com](https://domynswarm.domyn.com/)**.

| | |
| --- | --- |
| [Getting started](https://domynswarm.domyn.com/latest/getting-started/index.html) | install, launch a swarm, write your first custom job |
| [Guides](https://domynswarm.domyn.com/latest/guides/index.html) | Slurm and Lepton, checkpointing, sharding, data backends, monitoring |
| [Concepts](https://domynswarm.domyn.com/latest/concepts/index.html) | architecture, backend protocols, the `SwarmJob` lifecycle, configuration |
| [Reference](https://domynswarm.domyn.com/latest/reference/index.html) | CLI, configuration, environment variables, and Python API generated from source |

---

## Contributing

Issues and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers the development
setup, coding style, and commit conventions. Participation is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). To report a vulnerability, follow our
[security policy](SECURITY.md).

## License

The project uses the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Citation

```bibtex
@software{domyn_swarm,
  author  = {D'Ambrosio, Federico and Rognoni, Alessandro},
  title   = {domyn-swarm: LLM serving endpoints and high-throughput batch jobs on Slurm and DGX Cloud Lepton},
  year     = {2026},
  url     = {https://github.com/DomynAI/domyn-swarm},
  license = {Apache-2.0}
}
```

## Acknowledgements

Built on [vLLM](https://github.com/vllm-project/vllm) and [Ray](https://github.com/ray-project/ray),
with optional support for [NVIDIA DGX Cloud Lepton](https://www.nvidia.com/en-us/data-center/dgx-cloud-lepton/)
through the Lepton SDK.
