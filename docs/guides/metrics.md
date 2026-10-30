# Metrics and dashboards

domyn-swarm can run a Prometheus instance next to the load balancer, scrape
every vLLM replica, and give you a dashboard over the result.

This page covers tokens per second, queue depth, and GPU utilization. For health
failures, see [Monitoring and troubleshooting](monitoring.md).

Monitoring is off by default and supports only Slurm. It uses sidecars on the
load-balancer node, which the Lepton backend does not have.

## Turning it on

```yaml
backend:
  type: slurm
  endpoint:
    monitoring:
      enabled: true
      prometheus_image: /path/to/prometheus.sif
      nginx_exporter_image: /path/to/nginx-prometheus-exporter.sif
      retention: 12h
```

`enabled` is the master switch. When it is off, the load balancer ignores all
other monitoring fields.

Container mode runs Prometheus and `nginx-prometheus-exporter` as two sidecars.
The exporter converts the Nginx status page into metrics. To use host binaries
instead of containers:

```yaml
monitoring:
  enabled: true
  mode: binary
  prometheus_binary: /path/to/prometheus
  nginx_exporter_binary: /path/to/nginx-prometheus-exporter
```

Both binaries default to being looked up on `PATH` by name, so if they are
already installed cluster-wide, `mode: binary` alone is enough.

## What gets scraped

| Job | Target | Source |
| --- | --- | --- |
| `vllm` | every replica's `/metrics` | discovered from a file the supervisor keeps current |
| `nginx` | `nginx-prometheus-exporter` on `exporter_port` | static |
| `gpu` | each node's GPU exporter | only with `gpu_exporter.enabled` |
| `gpu_ownership` | `/gpu_ownership` on the endpoint | only with `gpu_exporter.enabled` |
| `ray` | each node's Ray metrics port | only with `ray_metrics.enabled` |
| `router` | the router's metrics endpoint | only when a router other than Nginx is enabled. See [Load balancing and routers](load-balancing.md#monitoring) |

Slurm determines each replica host and port after placement. Each replica writes
a `replica-<id>.head` file in the swarm serving directory. The load-balancer
  supervisor uses these files for Nginx upstreams and Prometheus targets. A new
  replica becomes a scrape target without manual configuration.

Every series is labeled `swarm` with the swarm's name, from Prometheus's
`external_labels`.

## Reaching it

Prometheus is served through the load balancer at `route_prefix` on the same
endpoint URL as the model:

```
http://<endpoint>/prometheus
```

The `port` field defaults to `9090` and controls the Prometheus listener on the
load-balancer node. Clients connect through the proxy instead of this port.

:::{warning}
CAUTION: Keep `/prometheus` inside the cluster. The route has no authentication,
so any endpoint client can read metrics from every replica.
:::

The database is local to the load-balancer node and ends with its job.
`retention` defaults to `12h` and limits only the current run. To keep metrics
after the swarm stops, use an external Prometheus server to federate the route.

The route uses the Prometheus HTTP API and works with any compatible viewer. See
[Viewing the metrics](#viewing-the-metrics).

## Viewing the metrics

The service uses the standard Prometheus API. Any compatible client can read it.
You can use Grafana or the terminal client.

### Your own Grafana

Add the swarm's `/prometheus` URL as a Prometheus data source in an existing
Grafana and build or import whatever dashboards you like. The dashboards
domyn-swarm bundles are ordinary Grafana JSON, so you can import them directly:

| Dashboard | Package path |
| --- | --- |
| vLLM and Nginx | `domyn_swarm/data/dashboards/vllm.json` |
| GPU, nvidia_smi vocabulary | `domyn_swarm/data/dashboards/gpu_nvidia_smi.json` |
| GPU, DCGM vocabulary | `domyn_swarm/data/dashboards/gpu_dcgm.json` |
| Ray panels, appended to another dashboard | `domyn_swarm/data/dashboards/ray_panels.json` |

Use this option when Grafana already runs in the cluster or when dashboards must
outlive a swarm. The endpoint is available only inside the cluster. Federate
metrics that must persist after the load-balancer job ends.

### In the terminal

```bash
domyn-swarm monitor my-swarm-name
```

This command resolves the Prometheus URL from the swarm state. It passes the URL
and bundled vLLM dashboard to
[grafatui](https://github.com/fedexist/grafatui), a terminal interface for
Prometheus. The command replaces the current process with grafatui.

grafatui is a separate project and is not installed with domyn-swarm. Its
[installation guide](https://fedexist.github.io/grafatui/installation.html)
documents `brew`, `install.sh`, and `cargo` methods.

If grafatui is not on `PATH`, the command exits with status 127 and prints the
Prometheus URL. You can use this URL with Grafana.

The command also uses these exit codes:

| Exit | Meaning |
| --- | --- |
| `1` | Monitoring is not enabled. Enable it and deploy the swarm again. |
| `2` | `--gpu` was passed but no GPU exporter is configured, or `--dashboard` names a file that does not exist |
| `127` | `grafatui` is not on `PATH` |

Exit 1 requires a new deployment because the load-balancer job starts monitoring
services during submission.

#### grafatui flags

```bash
domyn-swarm monitor my-swarm --range 1h --step 15s
domyn-swarm monitor my-swarm -d ./my-dashboard.json
domyn-swarm monitor my-swarm --prometheus-url http://other-host:9090
domyn-swarm monitor my-swarm --var replicas=8
```

`--range` and `--step` pass straight through to grafatui. `--prometheus-url`
overrides the resolved URL, which is how you point at a swarm's Prometheus from
outside the cluster or through a tunnel.

#### Dashboard variables

`monitor` fills two variables in the bundled dashboard. `vllm_job` contains the
Prometheus job name, and `replicas` drives the Replicas down panel. Use repeated
`--var KEY=VALUE` arguments to override or add variables.

`monitor` does not fill these variables:

- `swarm`: Each swarm has a separate Prometheus server, so a swarm filter is
  redundant.
- `model`: vLLM uses the full resolved model path as the metric label. Pass the
  label with `--var` if the dashboard uses it.

## GPU metrics

Per-node GPU metrics come from a separate exporter, enabled underneath
`monitoring`:

```yaml
backend:
  endpoint:
    monitoring:
      enabled: true
      gpu_exporter:
        enabled: true
        kind: nvidia_smi
        image: /shared/images/gpu_exporter_nvidia_smi.sif
```

Then:

```bash
domyn-swarm monitor my-swarm-name --gpu
```

The command loads the bundled dashboard that matches the configured `kind`.

### Choosing a kind

`nvidia_smi` (the default)
: A small static binary that shells out to `nvidia-smi`. Portable, runs
  unprivileged anywhere `nvidia-smi` exists.

`dcgm`
: NVIDIA's Data Center GPU Manager, emitting the standard `DCGM_FI_*` series.
  It uses the 3.x line because 4.x aborts when run unprivileged.
  Driver-to-DCGM compatibility varies by site. Use this exporter when you need
  the standard metric names and your driver supports it.

Both exporters run without privileges inside the job GPU cgroup. They report
memory, utilization, power, temperature, clocks, and throttle reasons. Profiling
counters (`DCGM_FI_PROF_*`) require root and are not available.

### Images and modes

Build from the recipes shipped in `images/`:

```bash
sudo singularity build gpu_exporter_nvidia_smi.sif images/gpu_exporter_nvidia_smi.def
sudo singularity build gpu_exporter_dcgm.sif images/gpu_exporter_dcgm.def
```

The exporter follows `monitoring.mode`, and the two kinds do not support the same
modes. The configuration rejects invalid combinations at load time instead of
failing on the node:

- `nvidia_smi` with `mode: container` needs an explicit `gpu_exporter.image`.
  There is no default image. Omitting it raises *nvidia_smi
  container mode needs an explicit gpu_exporter.image*.
- `dcgm` only works with `mode: container`, because it is launched through
  `singularity exec`. Asking for `mode: binary` raises *dcgm exporter is only
  supported in container mode*.
- `dcgm` with `mode: container` and no image falls back to a public NVIDIA image.

`mode: binary` therefore means `kind: nvidia_smi`, with `binary` defaulting to
`nvidia_gpu_exporter` on `PATH`.

### Which replica owns which GPU

An exporter runs once per node, but a node can host several replicas. Each
replica records the UUIDs of its GPUs. The supervisor renders these UUIDs as a
join metric:

```
dswarm_gpu_owner{uuid="GPU-...", UUID="GPU-...", replica="0"} 1
```

The ownership metric includes both label spellings. The `nvidia_smi` exporter
uses `uuid`, while DCGM uses `UUID`. One series can join data from
either exporter.

Prometheus scrapes this from `/gpu_ownership` on the endpoint, served by the same
Nginx that fronts the model.

## Ray metrics

For multi-node replicas, Prometheus scrapes Ray `ray_*` metrics from every node.
This needs no configuration: `ray_metrics.enabled` resolves itself to true when
monitoring is on and the deployment requires Ray, and to false otherwise.

Set it explicitly to `false` to opt out, or `true` to force it on. An explicit
value is always respected, and after validation the field is never left unset.

```yaml
monitoring:
  enabled: true
  ray_metrics:
    enabled: false     # scrape vLLM but not Ray
```

When Ray metrics are active, `domyn-swarm monitor` adds Ray panels to the bundled
vLLM dashboard. Passing `--dashboard` disables this addition and uses the given
dashboard without changes.

`ray_metrics.port` (default `8090`) is Ray's `--metrics-export-port`. It is fixed
rather than ephemeral so that the per-node files Prometheus discovers have stable
contents.

## What monitoring does not do

- Monitoring does not replace health checks. Prometheus reports performance,
  while the watchdog reports health. See
  [Watchdog and collector](../concepts/watchdog-collector.md).
- Monitoring data does not persist after the load-balancer job unless an
  external service federates it.
- Monitoring does not deploy Alertmanager or alert rules.
- The metrics do not require `domyn-swarm monitor`. That command provides a
  terminal viewer for the Prometheus data.

## Full field reference

Every field, with types and defaults, is generated from the models:
`MonitoringConfig`, `GpuExporterConfig` and `RayMetricsConfig` in
[Configuration](../reference/configuration.md).
