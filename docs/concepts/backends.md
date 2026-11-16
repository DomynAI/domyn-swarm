# Serving vs compute backends

:::{note}
"Backend" has two meanings in domyn-swarm. Serving and compute backends control
where models and jobs run. A data backend controls how jobs read and write data.
See [Choosing a data backend](../guides/data-backends.md).
:::

## Why two protocols instead of one backend type

A single `Backend` abstraction must serve models and run processes. These tasks
have different interfaces and failure modes. Separate protocols provide these
benefits:

- Serving and compute are selected independently. A model served in
  one place can be driven by work executing somewhere else.
- A new readiness strategy changes only the serving backend. A new process
  launcher changes only the compute backend.
- Endpoint readiness and job success remain separate states. As a result,
  `domyn-swarm status` can report a healthy endpoint and a failed job at once.

Both are `typing.Protocol` definitions. A backend conforms through its methods
and does not need to inherit a base class.

## ServingBackend

Owns the model endpoint and its lifetime.

| Method | Responsibility |
| --- | --- |
| `create_or_update(name, spec, extras)` | Create or reconcile the endpoint and return a `ServingHandle` |
| `wait_ready(handle, timeout_s, extras)` | Block until the endpoint can serve |
| `ensure_ready(handle)` | Assert readiness for an already-created endpoint |
| `status(handle)` | Report a `ServingStatus`, carrying a `ServingPhase` |
| `delete(handle)` | Remove the endpoint |

`create_or_update` reconciles an existing swarm when you run `up` again. It does
not fail or create a duplicate swarm.

## ComputeBackend

Owns the processes that call the endpoint.

| Method | Responsibility |
| --- | --- |
| `submit(...)` | Start a job and return a `JobHandle` |
| `wait(handle, stream_logs=True)` | Block until completion and return a `JobStatus` |
| `cancel(handle)` | Stop a running job |
| `probe(handle)` | Report a `JobProbe`, which is how `job status --refresh` works |
| `default_python(cfg)` | Interpreter to run the job with |
| `default_image(cfg)` | Container image, where the platform uses one |
| `default_resources(cfg)` | Platform resource request |
| `default_env(cfg)` | Environment the job needs |

The four `default_*` methods keep platform details out of the submission path.
The caller does not need to know that Slurm uses a `venv_path` and Lepton uses a
Docker image. `DefaultComputeMixin` supplies the common implementations. Most
backends can use its `probe` unchanged.

## How the two platforms satisfy them

Slurm uses an array job of vLLM replicas and an Nginx load-balancer job. An HTTP
probe calls `/health` through the load balancer. Compute uses `srun` in the
allocation. `require_allocated_node` prevents work on the load-balancer node.
With that setting, a job step gets all the memory and the CPU count of the
allocation that submits it.

Lepton uses an endpoint that manages its own replicas. It polls deployment state
instead of an HTTP health endpoint. Compute uses a Lepton batch job.

The readiness strategies differ, but both implement the same protocol.

## Adding a platform

The protocols provide a small, closed list of methods. They do not require a
base class.

Each backend must still implement readiness, log retrieval, identifier mapping,
and cancellation. It also needs a configuration model with a `type`
discriminator. See `backends/serving/` and `backends/compute/` for examples.

For method sets, handle contracts, registration, and platform-specific features,
see
[Implementing a backend](../guides/implementing-a-backend.md).
