# Architecture

Serving a model and running work against it are two different problems, and
domyn-swarm keeps them separate. This boundary determines the structure of the
system.

## DomynLLMSwarm owns the lifecycle

{py:class}`~domyn_swarm.core.swarm.DomynLLMSwarm` is a context manager. Entering
it starts an endpoint and waits for readiness. Leaving it stops the endpoint
unless `delete_on_exit=False` keeps the allocation alive.

```python
with DomynLLMSwarm(cfg=cfg) as swarm:
    swarm.submit_job(
        job,
        run=JobRunSpec(input_path=Path("in.parquet"), output_path=Path("out.parquet")),
    )
```

It is a thin coordinator. The platform-specific work lives below it.

## Deployment pairs a serving backend with a compute backend

`Deployment` composes exactly two collaborators and nothing else:

- `ServingBackend` owns the model endpoint.
- `ComputeBackend` owns the processes that call that endpoint.

Its flow is `up(name, ctx)` → `wait_ready(timeout_s)` → `run(...)` →
`down(handle)`. It is also a context manager. A raised exception still deletes
the endpoint instead of leaving an allocation active.

The handle passed between those calls is a `ServingHandle`, carrying the endpoint
URL and whatever platform metadata the backend needs to find its resources again.

## What each side is responsible for

The serving side provides:

`create_or_update`
: create the endpoint, or reconcile an existing one, returning a `ServingHandle`

`wait_ready`
: block until the endpoint can serve requests

`status`
: report a `ServingStatus`, which is what `domyn-swarm status` surfaces

`delete`
: remove the endpoint

The compute side provides:

`submit`
: start a job, returning a `JobHandle`

`wait`
: block until the job finishes, returning a `JobStatus`

`cancel`
: stop a running job

`probe`
: report a `JobProbe` for a handle, which is how status refresh works

Compute backends also supply `default_python`, `default_image`,
`default_resources`, and `default_env`. A job submission does not need to name
values that the platform can infer. `DefaultComputeMixin` provides common
implementations.

Both are `Protocol` definitions rather than base classes. A backend satisfies
them through its methods. See
[Serving vs compute backends](backends.md).

## Jobs run beside the endpoint, not inside it

A job never runs in the serving container. The compute backend starts a separate
process. Slurm uses `srun`, and Lepton uses a batch job. The process receives
`ENDPOINT` and `MODEL` in its environment and calls the endpoint over HTTP.

As a result, `job submit-script` can run arbitrary Python. The endpoint handles a
`SwarmJob` and a submitted script in the same way. A new job type does not
require a change to the serving backend.

See [The SwarmJob lifecycle](swarmjob-lifecycle.md).

## State makes swarms outlive processes

Every swarm has a record in a local SQLite database. The record contains
deployment metadata, resource handles, and the configuration that created the
swarm. It also contains platform identifiers, node assignments, and the endpoint
URL.

That record is why `--name` works. The original `up` process has ended by the
time you run `job submit --name my-swarm`. The command restores the swarm from
state. `DomynLLMSwarm.from_state(name)` and `swarm list` read the same record.

See [Managing swarm state](../guides/swarm-state.md).

## The load balancer reconciles rather than being configured

Nginx needs the location of each replica. Slurm decides placement after
submission, and replicas appear one at a time. The load-balancer configuration
must converge as locations become available.

Each replica writes a `replica-<id>.head` file, holding its `host:port`, into the
swarm's shared serving directory. A supervisor process watches that directory
and regenerates `00-upstreams.conf` from the current files. It also writes the
replica list to `replicas.txt`. If monitoring is on, it writes Prometheus target
files too. A file addition or removal represents a replica change.

The supervisor writes files but does not reload Nginx. Nginx runs in a separate
Singularity instance with a private PID namespace. A process in another
container cannot signal the Nginx master. The load-balancer script watches the
generated file on the host. After a change, it runs `nginx -t` and then
`nginx -s reload` through `singularity exec instance://`. It does not load an
invalid configuration.

A router can run behind Nginx on `127.0.0.1`. Nginx serves the public endpoint
and sends requests to the router through its `llm` upstream. Each router has a
template fragment that defines three shell functions: start, stop, and apply a
changed replica list.

A second loop on the host watches `replicas.txt` and calls the apply function
after a change. vllm-router adds and removes workers through its API. ramjet
and the Dynamo frontend restart. The loop also reads `run/router.pid` and
restarts the router if its process exits.

Only Dynamo's fragment defines `router_registrations`. The function adds worker
registrations to the snapshot that the loop compares. If a worker registers
again, the loop calls the apply function even if the replica list stays the same.
See [Load balancing and routers](../guides/load-balancing.md).

The same reconcile loop generates Prometheus targets from the replica head
files. Prometheus can scrape a new replica without a manual configuration
change. See
[Metrics and dashboards](../guides/metrics.md).

## Health is reported, not inferred

Each replica has a watchdog that probes it. The watchdog reports to one
collector, which owns the health database for `domyn-swarm status`. See
[Watchdog and collector](watchdog-collector.md) for the reason for this design.

## Where to go next

- [Serving vs compute backends](backends.md): The two protocols
- [The SwarmJob lifecycle](swarmjob-lifecycle.md): From CLI to output file
- [Watchdog and collector](watchdog-collector.md): Health reporting processes
- [Configuration precedence](configuration.md): Sources and priority of values
