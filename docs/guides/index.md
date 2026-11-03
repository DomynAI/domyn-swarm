# Guides

Each guide covers one task. Read the guide that matches your task.

## Platforms

- [Running on Slurm](slurm.md): Singularity images, bind mounts, modules, and node selection
- [Load balancing and routers](load-balancing.md): Router selection, configuration, and session IDs
- [Running on Lepton](lepton.md): Endpoints, job configuration, secrets, and platform differences

## Running work

- [Submitting jobs](submitting-jobs.md): Input formats, job classes, and common flags
- [Checkpointing and resuming](checkpointing.md): Resume an interrupted run
- [Sharding and concurrency](sharding-concurrency.md): Parallelism and resume behavior
- [Choosing a data backend](data-backends.md): pandas, polars, and Ray

## Operating

- [Managing swarm state](swarm-state.md): The state database, `db` commands, and swarm inspection
- [Monitoring and troubleshooting](monitoring.md): Status output and unhealthy replicas
- [Metrics and dashboards](metrics.md): Prometheus, GPU exporters, and `domyn-swarm monitor`
- [Swarm pools](swarm-pools.md): Current limitations of the incomplete pool feature

## Extending

- [Implementing a backend](implementing-a-backend.md): Protocols, configuration,
  and platform-specific features

For *why* things are built this way rather than how to use them, see
[Concepts](../concepts/index.md). For exhaustive flag and field lists, see
[Reference](../reference/index.md).

```{toctree}
:hidden:

slurm
load-balancing
lepton
submitting-jobs
checkpointing
sharding-concurrency
data-backends
swarm-pools
swarm-state
monitoring
metrics
implementing-a-backend
```
