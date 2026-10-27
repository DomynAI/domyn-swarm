# Load balancing and routers

On Slurm, Nginx distributes requests across a swarm's vLLM replicas. By default,
it sends each request to the replica with the fewest open connections
(`least_conn`). It does not track the KV cache, which stores intermediate
results from previous tokens. When a multi-turn agent reaches a different
replica, that replica must recompute the shared conversation history.

A router behind Nginx can send later turns to the replica that holds the
conversation's KV cache. Domyn Swarm supports three routers: vllm-router, NVIDIA
Dynamo, and ramjet. Select one router per swarm in `backend.endpoint.router`.
Routers are available only on the Slurm backend.

## How requests flow

```text
client --> nginx, 0.0.0.0:<endpoint port> --> router, 127.0.0.1:<router port> --> replica 1
                                                                              --> replica 2
                                                                              --> ...
```

Nginx serves the public `port`, `/health`, `/nginx_status`,
Prometheus, the Ray dashboard, and `/gpu_ownership`. The router runs in the
load-balancer job, on the same node as Nginx. It listens only on `127.0.0.1`.
Nginx forwards requests under `/v1/` to the router, which selects a replica.

With a router enabled, Nginx returns 404 for paths outside `/v1/` and the
locations listed above. This includes router administration paths such as
vllm-router's `/add_worker` and vLLM paths such as `/tokenize` and `/version`.

With the default `type: nginx`, Nginx balances requests itself. Existing
configurations keep this behavior.

The load-balancer job starts the router after all replicas are ready and before
Nginx. If a replica restarts on another node, the job updates the router's
replica list.

The job checks the router process every 5 seconds. If the router exits, the
job logs `[lb] WARNING: <router> exited; restarting` and starts it
again with the current replica list. Nginx returns 502 until the router is
healthy. `domyn-swarm status` shows the router type next to the load
balancer.

`nginx_timeout` and `enable_proxy_buffering` apply to the connection from Nginx
to the router. If clients stream responses, set `enable_proxy_buffering: false`.
Nginx then forwards tokens as the router sends them.

The router `port` defaults to 9200. vllm-router and ramjet serve metrics on
`metrics_port`, which defaults to 9201. Dynamo serves metrics on `port`. These
ports must be free on the load-balancer node and must differ from the endpoint
`port`.

### API key

With `type: nginx`, each vLLM replica checks the API key, and Nginx forwards the
`Authorization` header unchanged.

With any other router, Nginx checks the API key itself:

- Domyn Swarm reads the key from `SINGULARITYENV_VLLM_API_KEY`, then
  `VLLM_API_KEY`, then `API_TOKEN`, in the same order as for the replicas.
- If a key is set, Nginx returns 401 unless the request carries
  `Authorization: Bearer <key>`.
- The match is exact and case-sensitive.
- The key can contain only letters, digits, and the characters `._~+/=-`. If
  it contains another character, the load-balancer job stops with an error.
- `/health` stays open without a key.

## Choosing a router

The measurements in this guide used Qwen3.8-27B on four vLLM replicas with
tensor parallelism 2, on two Leonardo A100 nodes. Each configuration ran three
times. The results compare median performance against Nginx `least_conn` on the
same workload. Unless a section says otherwise, the replicas ran vLLM 0.31.0.
Dynamo ran in separate Slurm jobs, so its results compare with the Nginx run of
the same workload on other nodes.

| Workload | Router | Measured result |
|---|---|---|
| Long agent trajectories with a growing shared history, such as Terminal-Bench-style rollouts or agents with long tool gaps | vllm-router `cache_aware`, or Dynamo 1.6.0 with session affinity or ThunderAgent. If the client cannot send a session ID, use ramjet. | All three reduced mean trajectory time by 31% to 35%. vllm-router and Dynamo had similar results. ramjet was within 1 to 2 percentage points. |
| A steady group of agents with a new agent starting as each one finishes | vllm-router `cache_aware`, followed by Dynamo with ThunderAgent and ramjet. | Turns per minute increased by 28.6% with vllm-router and 26.7% with ThunderAgent. The increases were 25.3% for ramjet and 20.1% for Dynamo soft session affinity. |
| Short agent rollouts, a few turns each | vllm-router `cache_aware` or Dynamo with session affinity. | Rollouts ran about 16% to 17% faster with vllm-router or Dynamo, and 10% faster with ramjet. |
| Agents that share replicas with bulk batch traffic | vllm-router `cache_aware` or Dynamo with ThunderAgent. | Agents ran about 13.5% faster with vllm-router or Dynamo, and 10% faster with ramjet. |
| One lead agent that starts subagents with their own context | Keep Nginx or the router used for the rest of the workload. | Results ranged from 6% faster to 7% slower than Nginx, within the variation between runs. |
| Single-turn work: GRPO batches, pass@k sampling, few-shot evaluations | Nginx. | Differences stayed within the variation between runs. |

If vllm-router and Dynamo perform equally, choose vllm-router. It works with
plain `vllm serve` replicas and adds less latency.

Do not use vllm-router `consistent_hash` for work without session IDs. It sends
all such requests to one replica.

## Session IDs

A trajectory is a sequence of agent turns. A session ID identifies one
conversation or trajectory. vllm-router and Dynamo use it to send each turn to
the same replica. ramjet does not require a session ID.

For each conversation:

- Generate a random session ID, for example `uuid.uuid4().hex`.
  vllm-router `cache_aware` compares IDs character by character. Structured IDs
  with long shared prefixes, such as `traj-<task>-<n>`, put unrelated sessions on one replica.
- Limit the ID to 256 bytes.
- Send the same ID on every turn of the trajectory.
- Use a separate ID for each trajectory, including each sample in a GRPO
  group or pass@k set.

Send the ID in both fields on every request to support all routing policies:

| Place | Read by |
|---|---|
| `X-Session-ID` header | vllm-router `consistent_hash`, vllm-router `program`, and Dynamo |
| `session_params.session_id` body field | vllm-router `cache_aware` |

On the last turn, also send the header `X-Dynamo-Session-Final: true`.
Dynamo with `policy: thunderagent` then releases the session. The other routers
ignore this header.

The replicas and every router accept the `session_params` field. Domyn Swarm
starts the Dynamo frontend with `DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS=1`,
because the frontend otherwise rejects unknown body fields.

This example sends one trajectory with the OpenAI Python client:

```python
import uuid

from openai import OpenAI

client = OpenAI(base_url="http://<endpoint>/v1", api_key="<api key>")


def chat_turn(messages, session_id, final):
    headers = {"X-Session-ID": session_id}
    if final:
        headers["X-Dynamo-Session-Final"] = "true"
    return client.chat.completions.create(
        model="<model>",
        messages=messages,
        extra_headers=headers,
        extra_body={"session_params": {"session_id": session_id}},
    )


session_id = uuid.uuid4().hex
user_turns = ["List the files in the repository.", "Run the tests."]
messages = []
for turn, content in enumerate(user_turns):
    messages.append({"role": "user", "content": content})
    final = turn == len(user_turns) - 1
    reply = chat_turn(messages, session_id, final)
    messages.append({"role": "assistant", "content": reply.choices[0].message.content})
```

With NeMo RL, send the trajectory ID from the rollout code.
Use both the `X-Session-ID` header and the `session_params.session_id` body field.
Create one ID per trajectory.

### Session IDs in SwarmJob

`MultiTurnChatCompletionJob` sends session IDs automatically:

- It creates a random ID at the start of each conversation.
- It sends the ID on every turn, in the header and in the body field.
- The last turn also carries `X-Dynamo-Session-Final: true`.
- If a conversation is retried as a whole, it starts again from the first turn
  with a new ID. Dynamo keeps the aborted session until it expires.
- Single-turn jobs send no session ID.

If `request_params` contains `session_params.session_id`, the job sends that
ID in the header and body of every conversation. All conversations in that
job then go to one replica.

In the job's `request_params`, `session_params` must be a mapping.

## Nginx

Nginx balances requests with `least_conn`. Clients need no additional fields or
headers. The measurements in this guide use Nginx as the baseline.

```yaml
backend:
  type: slurm
  endpoint:
    nginx_image: /shared/images/nginx.sif
    router:
      type: nginx
```

Omit `router` for the same result. To build the Nginx image, see [Running on
Slurm](slurm.md#building-the-singularity-images).

With an API key, each vLLM replica checks the key. Nginx forwards the
`Authorization` header unchanged.

## vllm-router

vllm-router selects replicas by session ID. It does not inspect message text.
Without a session ID, the `cache_aware` policy uses replica load. Replicas run
plain `vllm serve`. In the measurements, the router added 5 to 40 ms to the time
to first token.

### Getting vllm-router

Build the image from
[`examples/singularity_images/vllm-router.def`](https://github.com/DomynAI/domyn-swarm/tree/main/examples/singularity_images/vllm-router.def):

```bash
sudo singularity build vllm-router_0.1.16.sif examples/singularity_images/vllm-router.def
```

Set `image` to the `.sif` file.

For a binary installation, use a virtual environment on a shared filesystem.
Install the `vllm-router` package from PyPI. Set `binary` to its
`bin/vllm-router` executable.

Set exactly one of `image` and `binary`. If `binary` contains no slash, the
load-balancer node resolves it on `PATH`.

The tested version is vllm-router 0.1.16. PyPI publishes its wheels for Linux
x86_64 and aarch64 (manylinux 2.28).

### vllm-router command

With `image`, Domyn Swarm runs this command:

```bash
singularity exec --bind "$HOST_DIR" /shared/images/vllm-router_0.1.16.sif vllm-router \
    --host 127.0.0.1 --port 9200 \
    --worker-urls http://<replica 1> http://<replica 2> ... \
    --prometheus-host 127.0.0.1 --prometheus-port 9201 \
    --policy cache_aware \
    --api-key "$LB_API_KEY"
```

With `binary`, the executable replaces `singularity exec ... vllm-router`. The
`policy` selects the flags after `--prometheus-port`. Domyn Swarm appends `args`
after them and sets `env` for the router process. If a key is set, it adds
`--api-key`.

If a replica moves, Domyn Swarm calls the router's `/remove_worker` and
`/add_worker` endpoints. The router keeps running.

### vllm-router policy `cache_aware`

```yaml
backend:
  endpoint:
    router:
      type: vllm-router
      image: /shared/images/vllm-router_0.1.16.sif
      policy: cache_aware
```

Domyn Swarm passes `--policy cache_aware` by default. To reuse cached
history, send `session_params.session_id` in the request body.

Compared with Nginx:

- Mean trajectory time fell by 35.3% for Terminal-Bench-style agents
  and 32.7% for agents with long tool gaps.
- A steady group of agents completed 28.6% more turns per minute.
- Short rollouts ran 16.2% faster. Agents sharing replicas with GRPO
  bursts ran 13.5% faster.
- GRPO batches, pass@16 sampling, and few-shot evaluations stayed within 2% of Nginx.

### vllm-router policy `consistent_hash`

```yaml
backend:
  endpoint:
    router:
      type: vllm-router
      image: /shared/images/vllm-router_0.1.16.sif
      policy: consistent_hash
```

Domyn Swarm passes `--policy consistent_hash`. Send `X-Session-ID` on every
request. The `up` command logs a warning for this policy because every request
without a session ID goes to one replica.

On vLLM 0.29.0, Terminal-Bench-style agents ran 30.0% faster and short rollouts
ran 14.1% faster than Nginx. In the same measurement, `cache_aware` gave 30.6%
and 15.6%. Without session IDs, a GRPO batch took 192.7% longer and a few-shot
evaluation took 243.9% longer.

### vllm-router policy `program`

```yaml
backend:
  endpoint:
    router:
      type: vllm-router
      image: /shared/images/vllm-router_0.1.16.sif
      policy: program
```

Domyn Swarm enables cache-aware routing with program scheduling:

```bash
--policy cache_aware --enable-program-scheduling \
    --program-scheduling-config-json '{"program_scheduling_enable_key":"auto","global_queue":true}'
```

Send `X-Session-ID` on every request.

On vLLM 0.29.0, Terminal-Bench-style agents ran 30.1% faster and short rollouts
ran 14.0% faster than Nginx. GRPO batches and few-shot evaluations stayed within
2% of the Nginx baseline.

### vllm-router policy `round_robin`

```yaml
backend:
  endpoint:
    router:
      type: vllm-router
      image: /shared/images/vllm-router_0.1.16.sif
      policy: round_robin
```

Domyn Swarm passes `--policy round_robin`. Clients need no additional fields or
headers. This policy was not measured.

### API key with vllm-router

Nginx authenticates requests as described in [API key](#api-key).
vllm-router forwards the key to the replicas with `--api-key`, and the
replicas authenticate these requests again.

### Limits of vllm-router

vllm-router does not inspect message text. Without session IDs, `cache_aware`
does not improve cache reuse.

## Dynamo

Workers report their cached tokens through KV events. The NVIDIA Dynamo
frontend uses these reports to select a worker. Session affinity keeps a
conversation on one replica. Each replica runs a Dynamo worker,
`python -m dynamo.vllm`, in place of `vllm serve`. The workers serve no OpenAI API. The frontend runs in the
load-balancer job and serves the API behind Nginx.

In the measurements, the Dynamo frontend added 30 to 160 ms to the time to first
token. The other routers added 5 to 40 ms.

### Getting Dynamo

Each Dynamo release requires a specific vLLM version. The tested release,
Dynamo 1.6.0, requires vLLM 0.31.0. Domyn Swarm runs both the frontend and the
workers from the swarm's `image`.

To include Dynamo in the image:

1. Build the vLLM image:

   ```bash
   sudo singularity build vllm_0.31.0-cu129.sif docker://vllm/vllm-openai:v0.31.0-cu129
   ```

2. Build the Dynamo image in the same directory from
   [`examples/singularity_images/vllm-dynamo.def`](https://github.com/DomynAI/domyn-swarm/tree/main/examples/singularity_images/vllm-dynamo.def):

   ```bash
   sudo singularity build vllm-dynamo_1.6.0.sif examples/singularity_images/vllm-dynamo.def
   ```

3. Set the swarm's `image` to `vllm-dynamo_1.6.0.sif`, and leave `venv` unset.

The recipe installs `ai-dynamo` without its `vllm` extra and keeps the image's
vLLM and torch. It uses the image's package versions as pip constraints.
`pydantic`, `pydantic_core`, and `googleapis-common-protos` stay unpinned,
because ai-dynamo 1.6.0 needs other versions of them.

If image builds are unavailable, create a virtual environment over a plain vLLM
image:

1. On a login node, run
   [`examples/dynamo/install_dynamo_venv.sh`](https://github.com/DomynAI/domyn-swarm/tree/main/examples/dynamo/install_dynamo_venv.sh):

   ```bash
   examples/dynamo/install_dynamo_venv.sh 1.6.0 /shared/images/vllm_0.31.0-cu129.sif /shared/venvs/dynamo-1.6.0
   ```

2. Set `venv` to the virtual environment and `image` to the same vLLM image.

The script creates the virtual environment with the image's Python and
`--system-site-packages`. It applies the same constraints as the recipe. Every
node must mount the shared filesystem at the same path. Domyn Swarm binds the
virtual environment into the containers and runs `<venv>/bin/python`. Without
`venv`, Domyn Swarm runs `python3` from `image`, which must then contain Dynamo.

PyPI publishes `ai-dynamo` 1.6.0 as a pure Python wheel. Its `ai-dynamo-runtime`
1.6.0 dependency has wheels for Linux x86_64 and aarch64 (manylinux 2.28).
Docker Hub publishes `vllm/vllm-openai:v0.31.0-cu129` for linux/amd64 and
linux/arm64.

### Dynamo commands

Each replica runs a worker under the watchdog. For replica `<id>` on port
`<port>`, Domyn Swarm runs this command:

```bash
SINGULARITYENV_DYN_SYSTEM_PORT=<port> \
SINGULARITYENV_DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS='["generate"]' \
SINGULARITYENV_DYN_TCP_RPC_HOST=<node IPv4 address> \
SINGULARITYENV_DYN_FILE_KV=<swarm directory>/dynamo-kv \
SINGULARITYENV_PYTHONHASHSEED=0 \
srun ... singularity exec --bind <mounts>,/shared/venvs/dynamo-1.6.0 --nv --writable-tmpfs <image> \
    python3 /opt/watchdog.py ... -- \
    /shared/venvs/dynamo-1.6.0/bin/python -m dynamo.vllm --model <model> \
    --tensor-parallel-size <gpus per replica> <args> \
    --namespace <swarm> \
    --discovery-backend file --request-plane tcp --event-plane zmq \
    --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","topic":"","endpoint":"tcp://*:<20080 + id>"}'
```

- The worker's system server listens on the replica port. The watchdog and the
  readiness check probe its `/health`.
- `DYN_TCP_RPC_HOST` needs an IPv4 address for each node. Domyn Swarm resolves
  it with `getent ahostsv4`. If a node has none, the replica stops with an
  error.
- Each worker publishes KV events on `kv_events_port` plus its replica ID. The
  `round_robin` policy omits `--kv-events-config`.
- Workers register in a discovery store in the swarm directory.

The load-balancer job runs the frontend. With `policy: thunderagent` and a
`venv`, Domyn Swarm runs this command:

```bash
export PYTHONHASHSEED=0
export DYN_FILE_KV="$HOST_DIR/dynamo-kv"
export DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS=1
exec env -u DYN_SYSTEM_PORT singularity exec --bind "$HOST_DIR",/shared/venvs/dynamo-1.6.0,<package>/templates/routers/dynamo_thunderagent.yaml <image> \
    /shared/venvs/dynamo-1.6.0/bin/python -m dynamo.frontend --http-host 127.0.0.1 --http-port $ROUTER_PORT \
    --namespace <swarm> \
    --discovery-backend file --request-plane tcp --event-plane zmq \
    --router-mode kv --router-kv-events --router-session-affinity-mode soft --router-session-affinity-ttl-secs 1800 --router-policy-config <package>/templates/routers/dynamo_thunderagent.yaml \
    >>"$ROUTER_LOG" 2>&1
```

- `<package>` is the installed `domyn_swarm` package directory.
- `$HOST_DIR` is the swarm directory, and `$ROUTER_PORT` is the router `port`.
- Domyn Swarm binds a policy file only for a policy that uses one.
- `env -u DYN_SYSTEM_PORT` removes a value inherited from the submitting shell.
  The frontend then serves metrics only on its HTTP port.
- `PYTHONHASHSEED=0` on the frontend and on every worker makes them hash KV
  blocks the same way.
- The `policy` selects the flags after `--event-plane zmq`. Domyn Swarm appends
  `args` after them and sets `env` for the frontend process.

### Dynamo policy `kv_session_affinity`

```yaml
backend:
  endpoint:
    router:
      type: dynamo
      venv: /shared/venvs/dynamo-1.6.0
      policy: kv_session_affinity
      session_affinity_ttl_s: 1800
```

Domyn Swarm uses this policy by default and passes
`--router-mode kv --router-kv-events --router-session-affinity-mode soft --router-session-affinity-ttl-secs <ttl>`.
The `session_affinity_ttl_s` field sets `<ttl>`, the time in seconds before an
idle session loses its binding to a replica. The default is 1800.

Send `X-Session-ID` on every request. No end-of-session signal is necessary.

Compared with Nginx:

- Terminal-Bench-style agents ran 35.1% faster, and agents with long tool
  gaps ran 32.8% faster.
- A steady group of agents completed 20.1% more turns per minute.
- Short rollouts ran 17.3% faster. Agents sharing replicas with GRPO
  bursts ran 13.1% faster.
- GRPO batches, pass@16 sampling, and few-shot evaluations stayed within 5% of Nginx.

### Dynamo policy `thunderagent`

```yaml
backend:
  endpoint:
    router:
      type: dynamo
      venv: /shared/venvs/dynamo-1.6.0
      policy: thunderagent
```

Domyn Swarm uses the `kv_session_affinity` flags and adds
`--router-policy-config <policy file>`. Send `X-Session-ID` on every turn. On
the last turn, also send `X-Dynamo-Session-Final: true`.

Compared with Nginx:

- Terminal-Bench-style agents ran 35.2% faster, and agents with long tool
  gaps ran 32.7% faster.
- A steady group of agents completed 26.7% more turns per minute.
- Short rollouts ran 16.6% faster. Agents sharing replicas with GRPO
  bursts ran 13.4% faster.
- GRPO batches, pass@16 sampling, and few-shot evaluations stayed within 5% of Nginx.

ThunderAgent performed at least as well as `kv_session_affinity` on every
measured agent workload. If the client sends `X-Dynamo-Session-Final`, use
ThunderAgent. Without that header, a session stays bound until
`session_affinity_ttl_s` expires.

Domyn Swarm ships this policy file and passes it to the frontend:

```yaml
request_classifier:
  type: thunderagent
worker_selection:
  aggregated: thunderagent
  instances:
    - name: thunderagent
      type: thunderagent
```

To replace the shipped policy file, set `policy_config` to another file's path.
The file also applies to `kv_session_affinity`. The `kv` and `round_robin`
policies ignore it.

### Dynamo policy `kv`

```yaml
backend:
  endpoint:
    router:
      type: dynamo
      venv: /shared/venvs/dynamo-1.6.0
      policy: kv
```

Domyn Swarm passes `--router-mode kv --router-kv-events`. Clients need no
additional fields or headers. This policy gave about half the improvement of
session affinity.

Compared with Nginx:

- Terminal-Bench-style agents ran 16.3% faster, and agents with long tool
  gaps ran 16.4% faster.
- A steady group of agents completed 9.6% more turns per minute.
- Short rollouts ran 5.4% faster. Agents sharing replicas with GRPO
  bursts ran 5.7% faster.
- GRPO batches, pass@16 sampling, and few-shot evaluations stayed within 5% of Nginx.

### Dynamo policy `round_robin`

```yaml
backend:
  endpoint:
    router:
      type: dynamo
      venv: /shared/venvs/dynamo-1.6.0
      policy: round_robin
```

Domyn Swarm passes `--router-mode round-robin`. The workers publish no KV
events. Clients need no additional fields or headers.

Agent trajectories took 1.9% to 15.8% longer than with Nginx. A steady group of
agents completed 4.9% fewer turns per minute.

### API key with Dynamo

Nginx authenticates requests to the public endpoint as described in
[API key](#api-key). The Dynamo frontend has no API key option, and the
workers serve no OpenAI API.

The API key does not protect a Dynamo swarm from hosts that reach the replica
nodes. The workers listen on the request plane, a TCP port on the node IPv4
address, and publish KV events on `kv_events_port` plus the replica ID. Neither
port checks a key. A host that reaches the request plane can send requests to
the workers without the key. A host that reaches a KV event port can read the
KV events.

### Tool calls and reasoning with Dynamo

With `type: dynamo`, Dynamo handles tool-call and reasoning parsing. Select
the parsers with Dynamo flags in `args`:

| vLLM flags | Dynamo flag |
|---|---|
| `--enable-auto-tool-choice --tool-call-parser <name>` | `--dyn-tool-call-parser <name>` |
| `--reasoning-parser <name>` | `--dyn-reasoning-parser <name>` |

This example selects the parsers for a Qwen3 model:

```yaml
args: >-
  --served-model-name model-name
  --enable-prefix-caching
  --dyn-tool-call-parser qwen3_coder
  --dyn-reasoning-parser qwen3
```

Dynamo returns `tool_calls` and `reasoning_content` in the same fields as vLLM
for both streaming and non-streaming responses.

Dynamo uses its own list of parser names, which can differ from vLLM names.
If a parser name is invalid, the worker exits at startup. The replica log
lists the valid names.

### Limits of Dynamo

- The frontend restarts when a replica joins, leaves, or registers again as
  a new worker instance after a watchdog restart. During the frontend restart,
  Nginx returns 502 for a few seconds.
- Dynamo does not support Ray mode, where one replica spans several nodes.
  If the swarm requires Ray, configuration validation rejects `type: dynamo`.
- Each node needs an IPv4 address.
- The replica `image` must hold the vLLM version that the Dynamo release pins.
- `args` go to `dynamo.vllm`. dynamo.vllm 1.6.0 exits on
  `--enable-auto-tool-choice` and on `--tool-call-parser`, so configuration
  validation rejects both flags with `type: dynamo`. See
  [Tool calls and reasoning with Dynamo](#tool-calls-and-reasoning-with-dynamo).

## ramjet

ramjet selects replicas using fingerprints, compact representations of the
request body. Clients do not need to change their requests or send session IDs.
ramjet reads its replica list only when it starts.

### Getting ramjet

Build the image from
[`examples/singularity_images/ramjet.def`](https://github.com/DomynAI/domyn-swarm/tree/main/examples/singularity_images/ramjet.def):

```bash
sudo singularity build ramjet_0.8.0.sif examples/singularity_images/ramjet.def
```

The recipe uses `ghcr.io/helixml/ramjet:v0.8.0`. The image is distroless and has
no shell. Domyn Swarm starts it with `singularity run`. The tested version is
ramjet 0.8.0. Its image is published for linux/amd64 only, and its GitHub
release has no binaries.

### ramjet command

With `image`, Domyn Swarm runs this command:

```bash
RJ_UPSTREAM=http://<replica 1>,http://<replica 2>,... \
RJ_API_ADDR=127.0.0.1:9200 \
RJ_METRICS_ADDR=127.0.0.1:9201 \
RJ_AFFINITY=prefix \
RJ_UPSTREAM_TOKEN="$LB_API_KEY" \
singularity run --bind "$HOST_DIR" /shared/images/ramjet_0.8.0.sif
```

With `binary`, the executable replaces `singularity run ...`. If a key is set,
Domyn Swarm sets `RJ_UPSTREAM_TOKEN`. It sets `env` for the router process and
appends `args` to the command.

### ramjet policy `prefix`

```yaml
backend:
  endpoint:
    router:
      type: ramjet
      image: /shared/images/ramjet_0.8.0.sif
      policy: prefix
```

Domyn Swarm sets `RJ_AFFINITY=prefix` by default. Clients need no additional
fields or headers.

Compared with Nginx:

- Terminal-Bench-style agents ran 33.7% faster, and agents with long tool
  gaps ran 31.1% faster.
- A steady group of agents completed 25.3% more turns per minute.
- Short rollouts ran 10.1% faster. Agents sharing replicas with GRPO
  bursts ran 9.9% faster.
- GRPO batches, pass@16 sampling, and few-shot evaluations stayed within 2% of Nginx.

`RJ_ROUTE_AFFINITY_BASIS=marginal` matched the default in a measurement on vLLM
0.29.0. To use it, set it through `env`.

### ramjet policy `load`

```yaml
backend:
  endpoint:
    router:
      type: ramjet
      image: /shared/images/ramjet_0.8.0.sif
      policy: load
```

Domyn Swarm sets `RJ_AFFINITY=load`. ramjet then selects replicas by load alone.
Clients need no additional fields or headers. This policy was not measured.

### API key with ramjet

Nginx authenticates requests as described in [API key](#api-key).
ramjet forwards the key to the replicas in `RJ_UPSTREAM_TOKEN`, and the
replicas authenticate these requests again.

### Limits of ramjet

ramjet restarts whenever the replica list changes because it reads that list
only at startup. During the restart, Nginx returns 502 for a few seconds.

## Processor architectures

An image or binary must match the node's processor architecture. Leonardo and
MareNostrum 5 use x86_64 nodes. JUPITER uses Grace Hopper nodes, which are
aarch64.

| Component | Tested version | Published architectures |
|---|---|---|
| vllm-router | 0.1.16 | x86_64 and aarch64 wheels on PyPI |
| ramjet | 0.8.0 | linux/amd64 image on GHCR |
| Dynamo (`ai-dynamo-runtime`) | 1.6.0 | x86_64 and aarch64 wheels on PyPI |
| vLLM image for Dynamo | 0.31.0 (`v0.31.0-cu129`) | linux/amd64 and linux/arm64 on Docker Hub |

## Monitoring

With monitoring on, Prometheus scrapes the router as job `router`:

| Router | Target | Path |
|---|---|---|
| vllm-router | `127.0.0.1:<metrics_port>`, default 9201 | `/metrics` |
| ramjet | `127.0.0.1:<metrics_port>`, default 9201 | `/metrics` |
| Dynamo | `127.0.0.1:<port>`, the frontend port, default 9200 | `/metrics` |

With `type: nginx`, Prometheus has no `router` job. Router selection does not
change the `vllm`, `nginx`, or GPU jobs. See [Metrics and dashboards](metrics.md).
