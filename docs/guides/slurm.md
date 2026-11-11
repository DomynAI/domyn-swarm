# Running on Slurm

On Slurm, a swarm is a job array of vLLM replicas fronted by an Nginx load
balancer, with roles assigned by `SLURM_NODEID`.

## What `up` submits

```bash
domyn-swarm up -c config.yaml --replicas 3
```

- `-c/--config`: Path to the YAML configuration
- `-r/--replicas`: Replica count override

Three things happen:

1. An array job starts one vLLM server per replica.
2. A load-balancer job starts Nginx. Nginx waits for every replica before it
   exposes one endpoint.
3. A SQLite state record lets later commands address the swarm by name.

## Building the container images

domyn-swarm supports Apptainer and SingularityCE on Slurm. Jobs select
`apptainer` when it is available on the execution node and otherwise fall back
to `singularity`. Build the two required images from the definition files in
[`examples/singularity_images/`](https://github.com/DomynAI/domyn-swarm/tree/main/examples/singularity_images),
on a machine with `sudo`, or with `--fakeroot` if your site enables it:

```bash
# NGINX load balancer image
sudo apptainer build nginx.sif examples/singularity_images/nginx.def

# vLLM runtime image
sudo apptainer build vllm.sif examples/singularity_images/vllm.def
```

The equivalent `singularity build` commands produce compatible images.

Then add the image paths to the configuration:

```yaml
model: "deepseek-ai/DeepSeek-R1-0528"  # whatever model you want to deploy
image: /shared/images/vllm.sif         # vLLM container; optional with backend.vllm_venv
backend:
  type: slurm
  endpoint:
    nginx_image: /shared/images/nginx.sif  # required for the load balancer
```

Practical notes:

- Put the `.sif` files on a shared path readable by every compute node.
- Apptainer or SingularityCE must be available on the execution nodes.
- If your site disables `--fakeroot`, build with admin privileges elsewhere and
  copy the `.sif` onto the shared filesystem.
- `image` is optional if you run vLLM from a virtual environment instead of a
  container, as described in
  [Running vLLM without a container](#running-vllm-without-a-container).
  `endpoint.nginx_image` is always required.

## Running vLLM without a container

`backend.vllm_venv` runs the replicas directly on the compute nodes from a
virtual environment with vLLM installed, instead of inside `image`:

```yaml
model: /shared/models/my-model
backend:
  type: slurm
  vllm_venv: /shared/venvs/vllm  # must contain bin/vllm
  endpoint:
    nginx_image: /shared/images/nginx.sif
```

With `vllm_venv` set:

- Each replica runs `vllm serve` with `<vllm_venv>/bin` first on `PATH`, so
  tools that vLLM calls at runtime, such as `ninja`, come from the same
  environment.
- The replica watchdog and the load balancer's collector and supervisor run
  with the environment's Python, from the same `domyn_swarm` installation
  paths that container mode bind-mounts.
- Nginx still runs from `endpoint.nginx_image`.
- `image` is ignored, so a site-wide default image does not need unsetting.
  `mounts` and `system_mounts` do not apply: the replicas see the node's own
  filesystem.
- Set `HF_HOME` in `env`: as in container mode, the replica job exports it
  from there, empty when unset.

The environment must sit on a filesystem every compute node can read, and its
Python and CUDA build must match the nodes. Before submitting, `up` checks that
`<vllm_venv>/bin/vllm` exists. Replicas larger than one node are not
supported: they need `image`.

## Path validation before submission

`up` examines configuration paths on the submit host before submission. If a path
cannot be used, the command does not submit the swarm:

```text
Config check failed: 1 path in swarm.yaml cannot be used

  image
      /shared/images/vllm.sif
      -> does not exist (deepest existing parent: /shared)

Nothing was submitted. Fix the paths above, or pass --skip-preflight to launch anyway.
```

The hint names the deepest path segment that exists. In this example, `images`
is missing. The command lists every invalid path and exits with status 2.

Checked: `image`, `model`, `endpoint.nginx_image`, the monitoring sidecar
images (container mode only), `mounts` sources and `venv_path`. Images can be a
`.sif` or an Apptainer or SingularityCE sandbox directory.

The preflight validation skips values that the submit host cannot resolve. These
values include registry references, Hugging Face repository IDs, bare names,
relative paths, unset variables, and paths under unreadable directories.
`system_mounts` uses per-node validation. See [Site mounts](#site-mounts). The validation
does not apply to Lepton.

```{note}
A `venv_path` that does not exist is now an error. Earlier versions ignored it
and used the ambient Python.
```

Use `--skip-preflight` when paths are visible only from compute nodes.
`job submit` and `job submit-script` also accept this flag. The Python API raises
`domyn_swarm.exceptions.SwarmConfigPathError`, and `DomynLLMSwarm` takes
`preflight=False`.

## Bind mounts

`backend.mounts` adds bind mounts to the vLLM containers:

```yaml
backend:
  type: slurm
  mounts:
    - /scratch/datasets                # bound at the same path inside the container
    - /host/config:/etc/app/config:ro  # host:container, with an option
```

Each entry is either `/path`, bound at the same path inside the container, or
`/host/path:/container/path` with an optional `:ro` or `:rw` suffix.

Entries pass unchanged to the selected runtime's `--bind`. domyn-swarm requires
an absolute source path and permits at most a `source:dest:opts` triple. The
container runtime reports errors such as a missing path or invalid option. For
the full bind syntax, see
[Apptainer](https://apptainer.org/docs/user/main/bind_paths_and_mounts.html) or
[SingularityCE](https://docs.sylabs.io/guides/latest/user-guide/bind_paths_and_mounts.html).

## Site mounts

Some clusters require host paths in every container for software loaded by
modules. For example, CINECA Leonardo uses `/leonardo/prod/opt`. Put site paths
in `backend.system_mounts`. Prefer the site
[defaults file](../reference/configuration.md) over each swarm configuration:

```yaml
# defaults.yaml
slurm:
  system_mounts:
    - /leonardo/prod/opt
```

```yaml
# or per deployment
backend:
  type: slurm
  system_mounts:
    - /leonardo/prod/opt
```

The syntax matches `backend.mounts`. A node binds an entry only when the host
path exists. Otherwise, the job log contains a warning. This behavior permits a
configuration to move between clusters. `/dev/infiniband` uses the same rule
without explicit configuration.

## Modules and sbatch preamble

Two fields inject site-specific setup into the generated cluster script:

```yaml
backend:
  type: slurm
  modules:
    - cuda/12.1
    - apptainer  # or the site's SingularityCE module
  preamble:
    - "#SBATCH --exclusive"
    - "export NCCL_DEBUG=WARN"
```

`modules` become `module load` lines. `preamble` lines appear before the module
loads. Use the preamble for extra sbatch directives or shell setup.

## Replicas larger than one node

A replica that needs more GPUs than one node becomes a Ray cluster. The cluster
has one head and enough workers to reach `gpus_per_replica`. vLLM tensor
parallelism spans these nodes.

This behavior follows from the arithmetic:

```yaml
model: "deepseek-ai/DeepSeek-R1-0528"
gpus_per_replica: 16     # more than one node holds
gpus_per_node: 4         # so each replica spans four nodes
replicas: 2
```

`gpus_per_replica > gpus_per_node` sets `requires_ray`. This value:

- Selects `llm_swarm_ray.sh.j2` instead of `llm_swarm.sh.j2`. Both templates use
  `_swarm_common.sh.j2` for shared behavior.
- Enables `watchdog.ray` to probe cluster liveness with the HTTP probe.
- Enables `monitoring.ray_metrics` when monitoring is on. The dashboard then
  includes Ray metrics and panels.

`gpus_per_replica` must be a multiple of `gpus_per_node` in this case.
Any other value leaves a partly used node in the replica and is rejected
with *When gpus_per_replica > gpus_per_node, gpus_per_replica must be a multiple
of gpus_per_node*.

`backend.template_path` selects a custom template and disables automatic
selection. A custom template must support Ray when the deployment requires it.

## Node selection and limits

`backend.partition`, `account` and `qos` are required. Beyond those,
`exclude_nodes` and `node_list` accept Slurm's own syntax
(`node[001-004]`), `time_limit` caps the allocation, and `mail_user` enables
END and FAIL notifications. The load-balancer job is configured separately under
`backend.endpoint`, including its own optional `qos` override.

`backend.requeue` defaults to `true` and emits `#SBATCH --requeue`. Replica jobs
then return after a node failure or preemption. The Ray head can also request a
requeue after a cluster failure. Set it to `false` on sites that forbid requeue:

```yaml
backend:
  type: slurm
  requeue: false
```

The directive is then omitted. A Ray head that loses its cluster exits non-zero
instead of calling `scontrol requeue`. The replica stays down instead of
restarting. This site property also belongs in `defaults.yaml` as
`slurm.requeue`.

Full field list: [Configuration](../reference/configuration.md).

## Shutting down

```bash
domyn-swarm down my-swarm-name
```

The command takes a swarm name. It uses `scancel` to stop the load balancer and
every replica job.

## Metrics

The load balancer can run Prometheus and a GPU exporter as sidecars. This
Slurm-only feature is off by default. See [Metrics and dashboards](metrics.md).
