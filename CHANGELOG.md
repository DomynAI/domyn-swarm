## v0.35.0 (2026-11-02)

### BREAKING CHANGE

- Python 3.10 is no longer supported.

### Fix

- **deps**: upgrade packages with open security advisories
- update logo
- **release**: bump install instructions with the package version

## v0.34.0 (2026-10-07)

### BREAKING CHANGE

- DomynLLMSwarm.submit_job requires run=JobRunSpec(...)
and no longer accepts flat run parameters. SwarmJob constructors raise
TypeError for names that are not configuration fields, including
kwargs={...}. Pass provider parameters as request_params={...}, also in
--job-kwargs.

### Feat

- support Python 3.14
- remove run and request parameters deprecated for 0.33
- **slurm**: size job steps from the submitting allocation

### Fix

- **preflight**: keep unreadable paths unreported on Python 3.14
- **slurm**: refuse job steps inside the load-balancer allocation
- **deps**: resolve 11 of 12 open Dependabot advisories

## v0.33.0 (2026-09-25)

### Feat

- **docs**: apply the Domyn brand theme to the documentation site
- **config**: check config paths before submitting a swarm

### Fix

- **deps**: override anyio to 4.14.2 for TLS and deadlock advisories

### Refactor

- align code and prose with development guidelines
- trim nonessential source comments

### Perf

- **slurm**: scale the nginx load balancer with the replica count

## v0.32.0 (2026-09-07)

### Feat

- **slurm**: make the watchdog collector port configurable
- **api**: export JobRunSpec from the domyn_swarm package root
- **jobs**: add JobConfig
- **execution**: select the engine with one parameter
- **execution**: add the shared sharded-execution pipeline
- **execution**: add the FrameOps adapters for pandas and Arrow

### Fix

- **slurm**: fail the lb job when the collector cannot start
- **jobs**: infer config_class from a class-level config object
- **execution**: stop dropping completed rows from polars shard output
- **execution**: stop wiping directory output on pandas global-resume
- **execution**: make --shard-output control sharded directory output
- **execution**: tolerate empty id-hash shards in the polars runner
- **checkpoint**: handle an empty result in InMemoryStore
- **execution**: tolerate empty shards in id-hash sharding
- **checkpoint**: keep the real id column on global resume

### Refactor

- **core**: submit_job takes a JobRunSpec
- **core**: move JobRunSpec out of the CLI layer
- **jobs**: build SwarmJob from JobConfig
- **execution**: move the pandas engine onto the shared pipeline
- **execution**: move the Arrow engine onto the shared pipeline

## v0.31.1 (2026-09-01)

### Feat

- **slurm**: add support for site-specific bind mounts in configuration

## v0.31.0 (2026-09-01)

### Feat

- **slurm**: add support for job requeueing configuration and handling

### Fix

- **deps**: resolve 39 of 40 open Dependabot advisories
- **tests**: fix pre-commit hooks
- **cli**: report the backend in `swarm list` instead of a blank column
- **state**: restore jobid validation lost in compute-backend extraction
- **config**: read settings at the point of use
- **watchdog**: make the inert watchdog config fields do something
- **jobs**: rename --num-threads to --num-shards
- **jobs**: reject misspelled SwarmJob parameters at construction
- **core**: release the allocation when startup fails
- **core**: report UNKNOWN from status() instead of raising
- **api**: export run_job_unified from the package root
- **docs**: publish a release's docs when its tag exists

### Refactor

- **state**: delete load_monitor_view workaround
- **core**: stop the constructor writing to disk
- **state**: build swarms from records instead of back-filling them
- **core**: expose serving_status() instead of reaching through privates
- **plan**: let the deployment plan build its own compute backend
- **watchdog**: align the runtime config with the YAML schema
- make jobs/runner.py re-export instead of reimplementing
- remove unreachable functions
- remove the superseded raw SQL query files
- remove the unused reverse_proxy helper

## v0.30.0 (2026-08-25)

### Feat

- **docs**: deploy through the Pages artifact instead of a gh-pages branch
- **docs**: add versioned gh-pages deployment
- **docs**: generate the CLI reference over the lazy command group
- **docs**: generate config and environment reference from models
- **monitor**: append Ray dashboard panels when Ray metrics are active
- **monitor**: add Prometheus ray scrape job and wire --emit-ray-targets
- **monitor**: export Ray metrics port and announce ray-*.target on all nodes
- **lb_supervisor**: emit ray_targets.json from per-node ray-*.target files
- **config**: add ray_metrics with auto-on resolution for Ray deployments
- **monitor**: launch per-node GPU exporter + ownership on Ray head and workers
- **config**: auto-select the Ray template when a deployment requires Ray
- **monitor**: guard unsupported GPU-monitoring configs (Ray, nvidia_smi/container, dcgm/binary)
- **images**: add nvidia_smi + dcgm GPU exporter build recipes
- **monitor**: --gpu flag loads the GPU dashboard for the configured kind
- **monitor**: bundle nvidia_smi + dcgm GPU dashboards (uuid->replica join)
- **monitor**: launch per-node GPU exporter + capture per-replica GPU ownership
- **monitor**: enable supervisor GPU target/ownership emission in lb.sh
- **monitor**: serve gpu_ownership.prom via nginx for Prometheus scrape
- **monitor**: add gpu + gpu_ownership Prometheus scrape jobs
- **data**: bundle DCGM core-only counters CSV (no profiling fields)
- **lb_supervisor**: emit gpu_targets.json + gpu_ownership.prom under --emit-gpu-targets
- **lb_supervisor**: render uuid->replica GPU ownership join metric
- **lb_supervisor**: render deduped per-node GPU exporter targets
- **config**: add GpuExporterConfig (off by default, nvidia_smi/dcgm kinds)
- **slurm**: add QoS override for load-balancer job and update template usage
- **lb_supervisor**: refactor head file reading and update render_targets to include replica labels
- **monitor**: add _pretty_argv function for improved argument display in grafatui launch
- **monitor**: implement lightweight monitor view for SwarmStateManager and update tests
- **config**: update model configuration and enhance monitoring settings
- **monitoring**: add range-average generation throughput (overall + per replica)
- **monitoring**: parameterize bundled dashboard with swarm/model/vllm_job/replicas vars
- **monitor**: support custom dashboards via --dashboard, drop --no-dashboard
- **cli**: add 'domyn-swarm monitor' command
- **data**: add bundled vLLM/nginx Grafana dashboard
- **config**: add MonitoringConfig to SlurmEndpointConfig
- **runtime**: add generation-only LB supervisor
- **runtime**: add lb_supervisor upstreams generator

### Fix

- **docs**: drop the tag trigger, which could never deploy
- **docs**: stop the navbar wordmark running under the nav links
- **docs**: publish at the Pages custom domain, not the github.io URL
- **cli**: preserve lazy subcommand hierarchy
- **monitor**: thread RAY_LOG_DIR into the Ray GPU-exporter fan-out so it logs correctly
- **config**: resolve ray_metrics.enabled to a concrete bool in all cases
- **lb_supervisor**: aggregate per-node gpu-owner files under the integer replica id
- **deps**: patch dependency vulnerabilities flagged by Dependabot
- **monitor**: serve gpu_ownership from the nginx-mounted serving path
- **lb_supervisor**: emit both uuid and UUID labels on dswarm_gpu_owner for dcgm+nvidia_smi joins
- **types**: fix pyright typing
- **monitoring**: make config test defaults-independent and dashboard multi-replica aware

### Refactor

- **cli**: remove reverse proxy option from 'up' command
- **templates**: split llm_swarm into vLLM-only + Ray, share _swarm_common
- **monitor**: simplify dashboard variable handling and update tests
- **slurm**: drive LB nginx via supervisor; add prometheus sidecars

### Perf

- **cli**: defer heavy imports off the swarm-load path
- **cli**: load heavy sub-apps lazily via TyperGroup
- **lepton**: probe the Lepton SDK lazily, not at import
- **io**: import pandas lazily in helpers/io

## v0.29.0 (2026-06-04)

### Feat

- **cli**: add --refresh to job status and positional job-id to job cancel
- enhance Slurm backend with log path handling and improved job probing
- implement job probing and status refresh for Slurm and Lepton backends
- add job listing and status commands with JSON output support
- implement JobBuilder for SwarmJob serialization and deserialization
- enhance job submission with normalized handles and status updates
- add job submission persistence and update fake state manager for testing
- **slurm**: enhance SlurmComputeBackend with external ID handling and FIFO support
- **jobs**: implement job management with CRUD operations and migration for jobs table

### Fix

- **build**: drop redundant force-include for data/queries
- update datetime import to use timezone.utc causing issues with python 3.10

### Refactor

- remove unused imports and functions from job.py
- update job submission CLI and introduce job control commands
- **tests**: simplify fake subprocess handling in delete tests

## v0.28.0 (2026-05-27)

### Feat

- add support for user-defined Singularity bind mounts in Slurm configuration
- add creation timestamp retrieval and enhance status output formatting

### Fix

- fix monotonic filename generation for Parquet shards to ensure correct merge order
- handle job status idempotency and process cleanup on cancellation
- add threading lock to ensure safe database upgrades
- handle nginx config validation and instance shutdown on error

## v0.27.0 (2026-05-18)

### Feat

- add global resume and shard mode options for job execution
- add shard_mode option in CLI
- implement stable sharding strategy for job execution with 'id' and 'index' modes
- **slurm**: update cancellation logic to terminate srun process group and remove pgid from job handle
- **slurm**: enhance job handling with process group management and improved wait/cancel functionality

### Fix

- ensure swarm.db schema is migrated when using the programmatic API
- fix global_resume issue with arrow and polars causing incomplete output
- ensure resources are merged correctly before job execution
- improve nginx config generation and validation in lb.sh
- actually fix existing server default for creation_dt column in swarm table
- update existing server default for creation_dt column in swarm table
- add table normalization for concatenation and corresponding tests
- implement large offset handling in ArrowShardStore and add corresponding tests

### Refactor

- implement lazy loading for logger and swarm components in CLI modules
- replace hash_pandas_object with hash_array for stable sharding
- normalization and concatenation fallback for Arrow tables in ArrowShardStore

## v0.26.1 (2026-01-28)

### Fix

- **arrow**: align ids and merge semantics
- **arrow**: include base outputs when finalizing shards

## v0.26.0 (2026-01-27)

### Feat

- **scripts**: add aliases for 'ds' and 'dswarm' to entry points
- **db**: add prune command to delete dirty swarm records and implement delete_records method
- **sharding**: add support for shard output in Polars runner, enabling one parquet file per shard
- **polars**: update output handling to support directory outputs and add tests for LazyFrame streaming
- **parquet**: add support for brace range and glob patterns in parquet_hash function
- **patterns**: implement brace range expansion for file patterns in I/O operations
- **job**: add checkpoint_tag to JobRunSpec and submission parameters
- **srun**: add default node count to srun command and update test for CLI execution
- **compat**: enhance checkpointing and sharded execution validation in Arrow and Polars jobs
- **srun**: add support for Slurm allocation checks and update command construction
- **jobs**: enhance job execution with direct shard output support and async flushing
- **ray**: implement Ray backend support with address configuration and validation
- **ray**: add schema extraction and job batching support to RayBackend
- **jobs**: implement iter_job_batches method for backend classes and add JobBatch data structure
- **jobs**: enhance Polars backend with lazy execution and checkpointing support
- **job**: add checkpoint tag option for job submission and processing
- **jobs**: implement Polars runner and enhance job execution with checkpointing support
- **jobs**: add support for optional id column in job submissions and processing
- **jobs**: wire arrow runner selection through CLI
- **arrow**: add arrow runner core and generic checkpoint store
- **jobs**: add in-memory checkpoint store and options to disable checkpointing and resume
- **backends**: enhance Polars backend write functionality and add tests
- **chat**: enhance reasoning content handling in chat completion jobs
- **cli**: add job submit flags for data backend
- **jobs**: run jobs with selected data backend
- **data-backends**: add backend registry and pandas/polars/ray backends
- **request**: introduce _request_kwargs method to filter request kwargs and update API calls
- **planning**: centralize plan building and normalize deployment resources
- **deployment**: introduce DeploymentContext for normalized deployment handling and refactor related methods
- **io**: add support for sharded parquet file saving in save_dataframe function
- **batching**: add progress hooks and refactor run method for BatchExecutor
- **cli**: show replica rows in status view
- **slurm**: wire watchdog args and surface replica hints
- **runtime**: add watchdog args builder and status helper
- enhance fingerprint computation with stable representation and update tests for checkpoint manager
- extend CheckpointManager to include input_col and enhance fingerprint validation
- add payload normalization function and corresponding tests for collector and watchdog
- enhance CheckpointManager with expected_output_cols validation and add tests
- add progress parameter to BatchExecutor.run method and update tests

### Fix

- **store**: update parquet file pattern to match all parquet files in directory
- **jobs**: improve backend type hinting and simplify checkpointing logic
- **dependencies**: reorganize Polars and Ray dependencies in pyproject.toml and uv.lock
- **run**: streamline output handling in run_job_unified and remove unnecessary conversions
- **checkpoint**: stabilize finalize() id column handling
- **swarm**: remove JOB_KWARGS from job details dictionary
- **slurm**: include DSWARM_AGENT_VERSION in SLURM job script

### Refactor

- **jobs**: update and deprecate jobs module
- **tests**: rename unused variables in watchdog tests for clarity
- **checkpoint**: make ParquetShardStore arrow-native
- replace deprecated transform method with transform_items in multiple job classes

## v0.25.0 (2026-01-08)

### Feat

- add support for custom job resources in SlurmComputeBackend and SrunCommandBuilder
- enhance load_dataframe and save_dataframe functions to support directory input and glob patterns

### Fix

- refactor parquet_hash to improve file handling and hashing logic
- improve resource handling in SlurmComputeBackend by refining argument construction
- reorder environment and mail user handling in SrunCommandBuilder
- update save_dataframe to allow writing parquet datasets directly to a directory
- remove redundant directory creation in open_db and ensure directories are created in lb.sh.j2

## v0.24.0 (2025-12-23)

### Feat

- conditionally mount local Ray logs based on backend requirements
- add logging of the last 200 dmesg entries to shared node logs
- improve Ray log synchronization with local and shared directories
- enhance Ray log management with node-specific directories and synchronization of internal logs

### Fix

- correct string formatting in job submission log message
- fix example config for qwen

## v0.23.0 (2025-12-12)

### Feat

- enhance nginx configuration and add job requeue functionality in Slurm scripts
- add readiness timeout and improve health check handling during startup in watchdog
- add integration tests for watchdog and collector functionality
- add watchdog collector and update slurm scripts accordingly
- add replica summary rendering to swarm status and introduce watchdog database path
- add fail reasons fetching to watchdog, display fail reasons during up command
- add watchdog agent to monitor and restart vllm in case of failures
- **cli**: add autoupgrade when executing any cli command
- **state**: add alembic migration and baseline, update SwarmStateManager to use SQAlchemy
- **state**: add orm and model for state management

### Fix

- enhance error handling in Slurm job submission and improve watchdog capacity checks
- improve error handling during swarm allocation cleanup and add logging for replica failures
- enhance logging in watchdog by consolidating status and exit information into structured JSON output
- improve error handling and logging in collector and watchdog; update lb.sh.j2 and llm_swarm.sh.j2 for better configuration management
- improve error handling in database read and remove unnecessary WAL setup in collector
- ensure exit code is not None for clean exit in watchdog restart test
- handle BadStatusLine exception in HTTP check and add restart logging in watchdog
- update collector and watchdog to use TCP instead of UDP for message communication
- improve error handling in upsert_status for SQLite operations
- increase SQLite connection timeout and add error handling for schema creation
- increase SQLite connection timeout to reduce "database is locked" issues
- increase SQLite connection timeout and improve error handling for PRAGMA settings
- update test_get_current_rev to use migration context and mock database engine
- ensure watchdog configuration exists and update Ray settings based on replica and node count
- update get_current_rev function to use SQLAlchemy engine for retrieving current database revision
- enable Ray in watchdog configuration and adjust script parameter syntax
- rename restart_max to max_restarts in watchdog configuration and update related scripts
- update python command to python3 for watchdog script execution
- fix checkpointing logic and add debug logging in JobRunner
- fix some minor issues
- fix various issues in watchdog
- update paths in deepseek_r1.yaml for consistency and clarity
- fix checkpointing logic and add debug logging in JobRunner
- autoupgrade to work when swarm db is not present

### Refactor

- format test_slurm_serving_backend.py

## v0.22.1 (2025-11-14)

### Fix

- fix MultiTurnChatCompletionJob returning the incorrect list of results
- fix remaining ruff issues after rules update
- use vllm_api_key environment variable for health checks when present
- **cli**: improve logging and UX flow for down command when --select or --all are not used

### Refactor

- format test files
- :art: reduce complexity of down command
- add new ruff rules
- use new ruff rules and update codebase accordingly

## v0.22.0 (2025-11-07)

### Feat

- add io_only output mode
- **jobs**: add output_mode to SwarmJob and update job runner accordingly
- add config.yaml to swarm directory
- add usage of modules and preamble from config in sbatch scripts
- **cli**: add possibility to down a swarm by the name contained in the config file
- **cli**: add describe command to display the configuration of a swarm

### Fix

- fix how _on_flush is called
- remove unneeded prints, fix io_only case for keeping columns
- fix issue with io_only output mode selecting on wrong df
- fix circular import caused by OutputJoinMode
- remove upper when setting environment variables
- add exports for environment variables, remove NCCL exports
- fix yaml serialization of config when persisting it
- remove RAY_CGRAPH_get_timeout set in template
- loading from state in a with statement won't instantiate a new swarm anymore
- **logs**: make sure that the name is printed only if no tty is detected
- **logs**: fix logging to enable the swarm name to stdout
- **cli**: print swarm name in stdout after up command
- fix wait_endpoint_s usage in lb.sh.j2
- fix using wrong qos in llm_swarm.sh.j2
- fix issue with slurm state not being correctly fetched
- use correct configuration ifor timeout in lb.sh.j2
- **cli**: add --force flag to down command and possibility to not specify the swarm name using the last created swarm

## v0.21.5 (2025-10-22)

### Fix

- fix usage of output_column_name  in _load_job
- fix issue with swarm not being cleaned up if managed using programmatic api
- fix issue with SwarmJob.results not being ignored when job is serialized

### Refactor

- **jobs**: update docs and tests according to the deprecation of output_column_name
- **jobs**: :wastebasket: deprecate output_column_name as input parameter for SwarmJob, in favour of output_cols
- update default value for checkpoint_dir in DomynLLMSwarm.submit_job to swarm_dir/checkpoints

## v0.21.4 (2025-10-20)

### Fix

- :bug: fix some bugs occuring in co-located replicas using vllm

## v0.21.3 (2025-10-17)

### Fix

- improva symlink creation by showing only relevant files
- fix issue with typo symlink path
- issue with symlink creation for singularity instance logs
- **logs**: fix various issues with ray usage with new log directory structure
- fix issue with serving spec not correctly propagated when running on slurm with requires_ray set to false
- fix issue where environment variables set in config were not propagated correctly
- **cli**: :bug: fix a bug where resources weren't cleaned up when ctrl-c-ing
- **imports**: fix issue with attributeerror raised on python 3.10

### Refactor

- **logs**: :technologist: improve usability of logs by using a sane dir tree

## v0.21.2 (2025-10-14)

### Fix

- **imports**: fix issue with circular imports on Python 3.10

## v0.21.1 (2025-10-14)

### Fix

- **cli**: fix issue with using the CLI in python 3.10

## v0.21.0 (2025-10-14)

### Feat

- **cli**: add domyn-swarm swarm list command to show available swarms in db
- **cli**: update status command for a more complete report of swarm status

### Fix

- **state**: fix issue with state queries not being loaded by CLI
- **templates**: fix issue where co-located replicas weren't deployed correctly

## v0.20.2 (2025-10-06)

## v0.20.1 (2025-10-03)

### Fix

- fix typing

## v0.20.0 (2025-10-03)

### BREAKING CHANGE

- swarms are now labelled with their deployment names
- db schema changed
- the name is now specified in the config file and then used to generate the primary key
- state is now managed with a sqlite db, state files are not needed anymore. In the CLI, state file paths are replaced by the job id

### Feat

- add usage of api_token when
- add usage of vllm_api_key environment variable propagated to vllm on singularity
- add status method for Swarm and related slurm implementation
- **state**: adapt cli to the new state management
- **state**: implement new state management
- **state**: change db schema
- unify swarm name handling
- implement init defaults command
- **docs**: add apache 2.0 badge
- **docs**: add domyn-swarm logo
- add settings to enable usage of environment variables
- add lepton api token support using lepton secrets, fix lepton job submission
- add endpoint submission to lepton
- start refactoring for job execution improvements
- **state**: add state management tests
- **state**: replace state management with a sqlite db
- **exceptions**: add custom exceptions
- **state**: add state management queries
- **ci**: add python version matrix

### Fix

- use correct fallback for LeptonConfig.lepton_workspace_id
- set home_directory attribute using the settings variable DOMYN_SWARM_HOME
- fix tests and wrong swarm name being displayed in logs
- fix issue with api_token not being used and passed to lepton jo
- **lepton**: fix lepton deployment by passing the correct objects and setting the proper image from the config
- update submit_script method
- fix issue with multiple output columns not handled correctly
- remove --platform flag from cli commands
- fix job submission by using correct venv_path attribute in slurm config and by making headers to openai client optional
- **linting**: remove unused imports
- **state**: fix conflicts
- **state**: fix handle loading
- **cli**: change swarm name help
- **cli**: change job name help
- **tests**: fix pool test
- fix type checking
- **state**: add newline
- **state**: improve docstrings
- **tests**: fix state tests and add new ones
- **state**: change slurm backend validation
- **tests**: fix broken tests
- **imports**: fix leptonai imports
- fix issue when validation not working when a required field is not present in the defaults
- fix issues related to lazy imports in tests
- **logo**: use relative paths
- **logo**: remove spaces
- **logo**: add both media
- **logo**: use raw links
- **logo**: add white logo for dark themes
- **logo**: add selector to invert colors
- **logo**: remove background
- **logo**: try to invert colors
- **logo**: add white background
- fix SlurmConfig not being actually passed to SlurmComputeBackend construction
- fix test
- fix test in TestSwarmStateManager
- fix model validator for removed backends property
- fix tests after refactoring
- fix imports and use proper propagation of secret for endpoint
- fix lepton endpoints and jobs deployments
- fix missing persist while waiting for endpoints
- **state**: add newlines to comply with hooks
- **reverse_proxy**: add support for older python versions

### Refactor

- add deprecation warnings for implemented SwarmJobs, implement new api
- **cli**: replace submit_app with job_app
- implement lazy imports for leptonai imports, which are extras
- add missing file in previous commit
- use a single backend per file, in place of a list
- fix templates and update example configs
- update package structure
- :boom: add new backends implementation for configuration, supporting multiple backends in a single config
- use Deployment class to handle the deployment of the compute jobs
- add new generic platform readiness class to abstract health checks

## v0.15.0 (2025-09-10)

### Feat

- **readme**: add badges

### Fix

- fix issue with environment variables not being expanded in config

## v0.14.0 (2025-08-29)

### Feat

- **cli**: add --checkpoint-dir to CLI

### Fix

- revert vllm_use_v1 to 0
- fix issue with checkpoint dir not being propagated correctly
- **tests**: fix tests for chat completion job

### Refactor

- fix pre-commit hooks execution

## v0.13.1 (2025-08-26)

### Refactor

- add parse_reasoning parameter to ChatCompletionJob

## v0.13.0 (2025-08-26)

### Feat

- execute hooks during the ci pipeline
- add pre-commit hooks
- add preliminary ci pipeline
- update cuda, vllm version and add ray dashboard

### Fix

- remove --contain flag to vllm instance deployment
- switch to python 3.12 in the ci pipeline
- reformat project with ruff
- Delete unused import
- simplify test to ease ci
- move ci file into the workflow dir to trigger the pipeline
- remove unneeded None guard
- fix obsolete tests
- enforce pyright type checking

## v0.12.0 (2025-08-19)

### Feat

- :zap: implement replicas sharing nodes to optimize resource usage

### Fix

- remove typing import not compatible with python 3.10
- fix edge case where multiple vllms wouldn't allocate on the same node
- fix issue with usage of min in jinja2 template
- fix how tensor-parallel-size is set when running vllm serve
- fix issue where gpus weren't allocated directly with --overlap flag

## v0.11.3 (2025-07-25)

### Fix

- add None defaults to deprecated parameters
- fix issue with parse args and improve logging
- add missing configuration key in DriverConfig model

### Refactor

- introduce checkpoint_interval and max_concurrency parameters to replace batch_size and parallel

### Perf

- add driver.nginx_timeout for load balancer config

## v0.11.2 (2025-07-18)

### Fix

- fix unescaped variable in nginx config
- add additional safeguards to timeouts in nginx config
- remove nginx_error.log
- :bug: fix issue with log folder not being created
- fix tests
- disable deletion of checkpoint files

## v0.11.1 (2025-07-18)

### Fix

- :bug: add error log for nginx and add lest_conn to load balancer config

## v0.11.0 (2025-07-17)

### Feat

- add enable_proxy_buffering option to driver config
- add --mail-user flag to submit job command

### Refactor

- add srun_builder module
- move state manager, slurm driver and lb health checker to their own modules in core package
- :construction: start refactoring DomynSwarm core logic
- :fire: remove unneeded LLMClient and related classes, keeping AsyncOpenAIClient
- :construction: refactor helpers module into its own package
- :construction: refactory jobs structure and its package

## v0.10.0 (2025-07-11)

### Feat

- add mail_user to configuration to enable mail notifications via slurm

### Fix

- enable log stats
- fix nodes allocation of ray workers

## v0.9.1 (2025-07-08)

### Fix

- fix issue with NOT_GIVEN used as default for openai timeout
- fix bug in batch execution progress bar
- fix status check for load balancer in domyn-swarm status

## v0.9.0 (2025-07-08)

### Feat

- add status command

### Fix

- improve logging
- fix issue with default template path

## v0.8.5 (2025-07-07)

### Fix

- fix issue with exceptions retries not being logged properly by tenacity

## v0.8.4 (2025-07-07)

### Fix

- use extra_body parameter in openai client to make sure to pass all kwargs to vllm

## v0.8.3 (2025-07-07)

### Fix

- fix issue after refactoring

### Refactor

- move config models to own package

## v0.8.2 (2025-07-07)

### Refactor

- refactor cli package structure

## v0.8.1 (2025-07-04)

### Fix

- fix issue with allocation of nodes for workers in swarms with multiple replicas

## v0.8.0 (2025-07-04)

### Feat

- implement pool command to spin up multiple clusters with different configs
- add --detach flag to domyn-swarm submit job

### Fix

- fix issue increasing wait time for ray workers
- add to_path helper
- fix progress bar batch request execution in last batch

## v0.7.0 (2025-07-03)

### Feat

- add swarm pool data models

### Fix

- improve robustness for checking slurm jobs state, add create_swarm_pool utility
- add PID returned if job is submitted as detached
- rmeove unneeded files
- add --exclusive to load balancer template
- add --exclusive to load balancer template
- use output path as part of checkpointing naming convention
- use model to differentiate between checkpoints on the same source dataset
- fix issue when domyn-swarm python script is running in sbatch script

## v0.6.0 (2025-07-02)

### Feat

- add MultiTurnChatCompletion job

### Fix

- use squeue instead of sacct for checking slurm job status, add reasoning_content key to MultiTurnChatCompletionJob

## v0.5.3 (2025-07-02)

### Fix

- fix issue with srun not being able to be executed inside lb node
- fix checkpointing
- fix example data to mirror actual implementation of ChatCompletionJob

## v0.5.2 (2025-07-01)

### Fix

- add --exclusive=user to job allocation in sbatch scripts
- fix exceptions silenced when retried, add --exclusive when submitting sbatch jobs

## v0.5.1 (2025-07-01)

### Fix

- fix ChatCompletionJob to properly handle array of messages

## v0.5.0 (2025-07-01)

### Feat

- add resource configuration for the driver/lb task
- add support for running jobs in multithreaded mode

### Fix

- fix usage of new cli argument for nthreads
- fix issue when users filter df inside transform
- fix typo
- actually fix streaming of stdout to terminal
- fix how log_directory config is after home_directory
- add cli parameters to domyn_swarm.run_job so that it's more usable
- use subprocess.PIPE for stderr for running srun commands
- fix properly the usage of a moddel saved in a local folder. If the cfg.model is a folder, then it will be mounted by the vllm containers
- quick fix to make sure that models saved in folders are actually readable by the vllm container
- update deepseek_r1.yaml

### Perf

- :zap: improve deepseek R1 performances

## v0.4.0 (2025-06-27)

### Feat

- add perplexitymixin to enable computation of perplexity by users
- :sparkles: enable tuple unpacking when when fn is called in SwarmJob.batched

### Fix

- typo introduced in previous commit
- :bug: fix a bug in jobs having multiple output columns
- :bug: remove delete_on_close=False incompatible with python 3.10
- start fixing reverse proxy, update example configs
- improve docs, fix an issue with job kwargs incorrectly handled

## v0.3.0 (2025-06-24)

### Feat

- add custom input and output columns, add deepseek r1 distill config, use vllm-only for serving models on a single node

### Fix

- add retry mechanism and update deepseek r1 config
- fix issues with checkpoints not happening, tune deepseek_r1 deployment, update example datasets

## v0.2.1 (2025-06-19)

### Fix

- **cfg**: fix default value of venv_path to None

## v0.2.0 (2025-06-18)

### Feat

- **cli**: :sparkles: add submit job and submit script commands
- **cli**: add up command to allocate the clusters without running any script
- **cfg**: add time_limit, exclude_nodes and node_list configurations
- add load balancer and proper replicas management for deployed clusters
- add replicas to config
- **deploy**: :bricks: add reverse proxy flag after cluster is deployed
- **cfg**: :zap: add new config keys
- **cli**: :sparkles: add cli interface with typer
- :rocket: initial commit

### Fix

- various fixes: remove driver_script from config, add home_directory key
- **jobs**: :bug: fix several bugs with launching jobs on already available clusters
- use proper conditionals when defining exclude_nodes and node_list
- **cfg**: rename instances to nodes in configuration and code
- **deploy**: :wrench: fix issue with nginx reverse proxy

### Refactor

- :beers: remove unneeded code
