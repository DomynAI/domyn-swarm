# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import domyn_swarm.cli.job as mod
from domyn_swarm.platform.protocols import JobHandle, JobStatus

runner = CliRunner()


def _mk_files(tmp_path: Path):
    in_path = tmp_path / "input.parquet"
    out_path = tmp_path / "output.parquet"
    in_path.write_bytes(b"PARQUET_MOCK")
    return in_path, out_path


def _parse_last_json_line(text: str) -> dict:
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines, "No CLI output lines found."
    return json.loads(lines[-1])


def test_submit_script_with_config_runs_in_context(mocker, tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: test-model\nname: test-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    swarm = mocker.MagicMock()
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    result = runner.invoke(
        mod.job_app,
        [
            "submit-script",
            str(script),
            "-c",
            str(tmp_path / "config.yaml"),
            "--",
            "foo",
            "bar",
        ],
    )

    print(result.stderr)

    assert result.exit_code == 0
    swarm.submit_script.assert_called_once()
    assert swarm.submit_script.call_args.kwargs["extra_args"] == ["foo", "bar"]


def test_submit_script_with_name_uses_from_state(mocker, tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")

    swarm = mocker.MagicMock()
    swarm.name = "my-swarm"
    swarm.submit_script.return_value = JobHandle(
        id="123.0",
        status=JobStatus.RUNNING,
        meta={"job_id": "job-1", "pid": 4321, "external_id": "123.0"},
    )
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    result = runner.invoke(
        mod.job_app,
        ["submit-script", str(script), "-n", "my-swarm", "--", "X", "Y"],
    )

    assert result.exit_code == 0
    swarm.submit_script.assert_called_once()
    assert swarm.submit_script.call_args.kwargs["extra_args"] == ["X", "Y"]
    payload = _parse_last_json_line(result.stdout)
    assert payload["command"] == "submit-script"
    assert payload["swarm"] == "my-swarm"
    assert payload["job_id"] == "job-1"
    assert payload["pid"] == 4321
    assert payload["external_id"] == "123.0"


def test_submit_script_mutual_exclusion(mocker, tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")

    result = runner.invoke(
        mod.job_app,
        [
            "submit-script",
            str(script),
            "-c",
            str(tmp_path / "config.yaml"),
            "-n",
            "my-swarm",
        ],
    )
    assert result.exit_code != 0


def test_submit_script_raises_without_config_or_name(mocker, tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")

    result = runner.invoke(mod.job_app, ["submit-script", str(script)])
    assert result.exit_code != 0
    assert "State is null" in str(result.exception)


@pytest.mark.parametrize("expand_paths", [False, True])
def test_submit_job_with_config_happy_path(mocker, monkeypatch, tmp_path: Path, expand_paths):
    in_path, out_path = _mk_files(tmp_path)
    input_arg, output_arg = str(in_path), str(out_path)
    if expand_paths:
        monkeypatch.setenv("JOB_DATA_DIR", str(tmp_path))
        input_arg, output_arg = (
            f"${{JOB_DATA_DIR}}/{in_path.name}",
            f"${{JOB_DATA_DIR}}/{out_path.name}",
        )
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    swarm = mocker.MagicMock()
    swarm.endpoint = "http://host:9000"
    swarm.model = "my-model"
    swarm.name = "cfg-swarm"
    swarm.submit_job.return_value = JobHandle(
        id="123.0",
        status=JobStatus.PENDING,
        meta={"job_id": "job-1", "pid": 4321, "external_id": "123.0"},
    )
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    job_obj = object()

    def _fake_build(job_class, job_kwargs, **kwargs):
        assert job_class == "domyn_swarm.jobs:ChatCompletionJob"
        assert isinstance(job_kwargs, str)
        assert kwargs["endpoint"] == swarm.endpoint
        assert kwargs["model"] == swarm.model
        _ = json.loads(job_kwargs)
        return job_obj

    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", side_effect=_fake_build)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            input_arg,
            "--output",
            output_arg,
            "-c",
            str(tmp_path / "cfg.yaml"),
            "--job-kwargs",
            '{"temperature":0.2}',
            "--checkpoint-interval",
            "16",
            "--max-concurrency",
            "8",
            "--retries",
            "2",
            "--timeout",
            "123",
            "--num-shards",
            "4",
            "--limit",
            "100",
            "--detach",
        ],
    )
    print(res.stderr)
    assert res.exit_code == 0

    swarm.submit_job.assert_called_once()
    call = swarm.submit_job.call_args
    assert call.args[0] is job_obj
    run = call.kwargs["run"]
    assert run.input_path == in_path
    assert run.output_path == out_path
    assert run.num_shards == 4
    assert run.limit == 100
    assert run.detach is True
    assert run.checkpoint_dir is None
    payload = _parse_last_json_line(res.stdout)
    assert payload["command"] == "submit"
    assert payload["swarm"] == "cfg-swarm"
    assert payload["job_id"] == "job-1"
    assert payload["pid"] == 4321
    assert payload["external_id"] == "123.0"


def test_submit_job_with_name_happy_path(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)

    swarm = mocker.MagicMock()
    swarm.endpoint = "http://endpoint"
    swarm.model = "m"
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    job_obj = object()
    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", return_value=job_obj)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-n",
            "my-swarm",
        ],
    )

    assert res.exit_code == 0
    swarm.submit_job.assert_called_once()
    assert swarm.submit_job.call_args.args[0] is job_obj


def test_submit_job_mutual_exclusion(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(tmp_path / "x.yaml"),
            "-n",
            "name",
        ],
    )
    assert res.exit_code != 0  # typer.Exit(1)


def test_submit_job_raises_if_no_config_and_no_name(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    res = runner.invoke(
        mod.job_app,
        ["submit", "--input", str(in_path), "--output", str(out_path)],
    )
    assert res.exit_code != 0
    assert "Swarm name is null" in str(res.exception)


def test_submit_job_keyboard_interrupt_abort(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    swarm = mocker.MagicMock()
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    cm.cleanup = mocker.MagicMock()
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    def _boom(*a, **k):
        raise KeyboardInterrupt()

    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", side_effect=_boom)

    mocker.patch.object(mod.typer, "confirm", return_value=True)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(tmp_path / "cfg.yaml"),
        ],
    )
    assert res.exit_code != 0  # Aborted
    print(res.stdout)
    print(res.stderr)
    cm.cleanup.assert_called_once()
    assert "Swarm allocation cancelled by user" in res.stdout


def test_submit_job_keyboard_interrupt_continue(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    swarm = mocker.MagicMock()
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    cm.cleanup = mocker.MagicMock()
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", side_effect=KeyboardInterrupt)
    mocker.patch.object(mod.typer, "confirm", return_value=False)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(tmp_path / "cfg.yaml"),
        ],
    )
    assert res.exit_code == 0
    cm.cleanup.assert_not_called()
    # Avoid being strict with ellipsis char
    assert "Continuing to wait for job to complete" in res.stdout


def test_submit_job_forwards_specific_options(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    swarm = mocker.MagicMock()
    swarm.endpoint = "http://host:9000"
    swarm.model = "m"
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    job_obj = object()
    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", return_value=job_obj)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(tmp_path / "cfg.yaml"),
            "--num-shards",
            "3",
            "--limit",
            "50",
            "--detach",
            "--mail-user",
            "me@example.com",
            "--checkpoint-dir",
            str(tmp_path / ".ckpt"),
        ],
    )

    print(res.output)

    assert res.exit_code == 0
    swarm.submit_job.assert_called_once()
    run = swarm.submit_job.call_args.kwargs["run"]
    assert run.num_shards == 3
    assert run.limit == 50
    assert run.detach is True
    assert run.mail_user == "me@example.com"
    assert run.checkpoint_dir == tmp_path / ".ckpt"


def test_submit_job_forwards_ray_address(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    cfg_obj = object()
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)

    swarm = mocker.MagicMock()
    swarm.endpoint = "http://host:9000"
    swarm.model = "m"
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)

    job_obj = object()
    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", return_value=job_obj)

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(config_path),
            "--data-backend",
            "ray",
            "--id-column",
            "doc_id",
            "--ray-address",
            "ray://head:10001",
        ],
    )

    assert res.exit_code == 0
    swarm.submit_job.assert_called_once()
    run = swarm.submit_job.call_args.kwargs["run"]
    assert run.ray_address == "ray://head:10001"


def _mock_config_swarm(mocker, *, backend_type: str):
    cfg_obj = SimpleNamespace(backend=SimpleNamespace(type=backend_type))
    mocker.patch.object(mod, "_load_swarm_config", return_value=cfg_obj)
    swarm = mocker.MagicMock()
    swarm.name = "my-swarm"
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    deploy = mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)
    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", return_value=object())
    return swarm, deploy


def _submit_args(tmp_path: Path, *extra: str) -> list[str]:
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")
    return [
        "submit",
        "--input",
        str(in_path),
        "--output",
        str(out_path),
        "-c",
        str(config_path),
        *extra,
    ]


def test_submit_job_forwards_job_step_resources(mocker, tmp_path: Path):
    swarm, _ = _mock_config_swarm(mocker, backend_type="slurm")

    res = runner.invoke(mod.job_app, _submit_args(tmp_path, "--mem", "64G", "--cpus-per-task", "8"))

    assert res.exit_code == 0, res.output
    run = swarm.submit_job.call_args.kwargs["run"]
    assert run.job_resources == {"mem": "64G", "cpus_per_task": 8}


def test_submit_job_without_resource_options_keeps_defaults(mocker, tmp_path: Path):
    swarm, _ = _mock_config_swarm(mocker, backend_type="slurm")

    res = runner.invoke(mod.job_app, _submit_args(tmp_path))

    assert res.exit_code == 0, res.output
    assert swarm.submit_job.call_args.kwargs["run"].job_resources is None


def test_submit_job_rejects_resource_options_on_lepton(mocker, tmp_path: Path):
    swarm, deploy = _mock_config_swarm(mocker, backend_type="lepton")

    res = runner.invoke(mod.job_app, _submit_args(tmp_path, "--mem", "64G"))

    assert res.exit_code != 0
    assert "resource_shape" in res.output
    deploy.assert_not_called()
    swarm.submit_job.assert_not_called()


def test_submit_job_with_name_forwards_job_step_resources(mocker, tmp_path: Path):
    in_path, out_path = _mk_files(tmp_path)
    swarm = mocker.MagicMock()
    swarm.name = "my-swarm"
    swarm.cfg.backend.type = "slurm"
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)
    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", return_value=object())

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-n",
            "my-swarm",
            "--mem",
            "0",
        ],
    )

    assert res.exit_code == 0, res.output
    assert swarm.submit_job.call_args.kwargs["run"].job_resources == {"mem": "0"}


def test_submit_script_forwards_job_step_resources(mocker, tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")
    swarm = mocker.MagicMock()
    swarm.name = "my-swarm"
    swarm.cfg.backend.type = "slurm"
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    res = runner.invoke(
        mod.job_app,
        ["submit-script", str(script), "-n", "my-swarm", "--cpus-per-task", "16"],
    )

    assert res.exit_code == 0, res.output
    assert swarm.submit_script.call_args.kwargs["job_resources"] == {"cpus_per_task": 16}


def test_submit_script_rejects_resource_options_on_lepton(mocker, tmp_path: Path):
    script = tmp_path / "script.py"
    script.write_text("print('hi')")
    swarm = mocker.MagicMock()
    swarm.cfg.backend.type = "lepton"
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    res = runner.invoke(
        mod.job_app, ["submit-script", str(script), "-n", "my-swarm", "--mem", "8G"]
    )

    assert res.exit_code != 0
    assert "resource_shape" in res.output
    swarm.submit_script.assert_not_called()


def test_wait_job_with_job_id_updates_status_and_emits_json(mocker):
    handle = JobHandle(id="123.0", status=JobStatus.RUNNING, meta={"job_id": "job-1"})
    target = mod.helpers.ResolvedJobTarget(
        swarm_name="my-swarm",
        handle=handle,
        job_id="job-1",
        source="job_id",
    )
    resolve = mocker.patch.object(mod.helpers, "resolve_job_target", return_value=target)
    emit = mocker.patch.object(mod.helpers, "emit_job_control_json")
    update = mocker.patch.object(mod.SwarmStateManager, "update_job")

    swarm = mocker.MagicMock()
    swarm.wait_job.return_value = JobStatus.SUCCEEDED
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    result = runner.invoke(mod.job_app, ["wait", "--job-id", "job-1"])

    assert result.exit_code == 0
    resolve.assert_called_once_with(
        job_id="job-1",
        external_id=None,
        handle_json=None,
        deployment_name=None,
    )
    swarm.wait_job.assert_called_once_with(handle, stream_logs=True)
    update.assert_called_once_with(
        "job-1",
        status=JobStatus.SUCCEEDED,
        external_id=None,
    )
    emit.assert_called_once()
    assert target.handle.status == JobStatus.SUCCEEDED


def test_cancel_job_with_external_id_sets_cancelled_and_updates_state(mocker):
    handle = JobHandle(
        id="123.0",
        status=JobStatus.RUNNING,
        meta={"job_id": "job-1", "external_id": "123.0"},
    )
    target = mod.helpers.ResolvedJobTarget(
        swarm_name="my-swarm",
        handle=handle,
        job_id="job-1",
        source="external_id",
    )
    resolve = mocker.patch.object(mod.helpers, "resolve_job_target", return_value=target)
    emit = mocker.patch.object(mod.helpers, "emit_job_control_json")
    update = mocker.patch.object(mod.SwarmStateManager, "update_job")

    swarm = mocker.MagicMock()
    swarm.cancel_job.return_value = JobStatus.CANCELLED
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    result = runner.invoke(
        mod.job_app,
        ["cancel", "--external-id", "123.0", "--name", "my-swarm"],
    )

    assert result.exit_code == 0
    resolve.assert_called_once_with(
        job_id=None,
        external_id="123.0",
        handle_json=None,
        deployment_name="my-swarm",
    )
    swarm.cancel_job.assert_called_once_with(handle)
    update.assert_called_once_with(
        "job-1",
        status=JobStatus.CANCELLED,
        external_id="123.0",
    )
    emit.assert_called_once()
    assert target.handle.status == JobStatus.CANCELLED


def test_cancel_job_with_positional_job_id(mocker):
    handle = JobHandle(
        id="123.0",
        status=JobStatus.RUNNING,
        meta={"job_id": "job-1", "external_id": "123.0"},
    )
    target = mod.helpers.ResolvedJobTarget(
        swarm_name="my-swarm",
        handle=handle,
        job_id="job-1",
        source="job_id",
    )
    resolve = mocker.patch.object(mod.helpers, "resolve_job_target", return_value=target)
    mocker.patch.object(mod.helpers, "emit_job_control_json")
    mocker.patch.object(mod.SwarmStateManager, "update_job")

    swarm = mocker.MagicMock()
    swarm.cancel_job.return_value = JobStatus.CANCELLED
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    result = runner.invoke(mod.job_app, ["cancel", "job-1"])

    assert result.exit_code == 0
    resolve.assert_called_once_with(
        job_id="job-1",
        external_id=None,
        handle_json=None,
        deployment_name=None,
    )
    swarm.cancel_job.assert_called_once_with(handle)


def test_list_jobs_json_output(mocker):
    rows = [
        {
            "job_id": "job-1",
            "deployment_name": "my-swarm",
            "provider": "slurm",
            "kind": "step",
            "status": "RUNNING",
            "raw_status": "RUNNING",
            "external_id": "123.0",
            "name": "my-job",
            "command": ["python", "-m", "x"],
            "resources": {"cpus_per_task": 4},
            "log_paths": {"stdout": "/tmp/out.log"},
            "error": None,
            "creation_dt": "2026-02-10T10:00:00",
            "update_dt": "2026-02-10T10:05:00",
        }
    ]
    list_jobs = mocker.patch.object(mod.SwarmStateManager, "list_jobs", return_value=rows)

    result = runner.invoke(
        mod.job_app,
        [
            "list",
            "-n",
            "my-swarm",
            "--status",
            "running",
            "--status",
            "cancelled",
            "--limit",
            "20",
            "--json",
        ],
    )

    assert result.exit_code == 0
    list_jobs.assert_called_once_with(
        "my-swarm",
        limit=20,
        statuses=[JobStatus.RUNNING, JobStatus.CANCELLED],
    )
    payload = _parse_last_json_line(result.stdout)
    assert payload["command"] == "list"
    assert payload["swarm"] == "my-swarm"
    assert payload["count"] == 1
    assert payload["statuses"] == ["RUNNING", "CANCELLED"]
    assert payload["jobs"][0]["job_id"] == "job-1"


def test_list_jobs_tui_uses_renderer(mocker):
    rows = [
        {
            "job_id": "job-1",
            "deployment_name": "my-swarm",
            "provider": "slurm",
            "kind": "step",
            "status": "RUNNING",
            "raw_status": None,
            "external_id": "123.0",
            "name": "my-job",
            "command": None,
            "resources": None,
            "log_paths": None,
            "error": None,
            "creation_dt": "2026-02-10T10:00:00",
            "update_dt": "2026-02-10T10:05:00",
        }
    ]
    mocker.patch.object(mod.SwarmStateManager, "list_jobs", return_value=rows)
    render = mocker.patch("domyn_swarm.cli.tui.job_view.render_job_list")

    result = runner.invoke(mod.job_app, ["list", "-n", "my-swarm"])

    assert result.exit_code == 0
    render.assert_called_once()


def test_status_job_json_output(mocker):
    row = {
        "job_id": "job-1",
        "deployment_name": "my-swarm",
        "provider": "slurm",
        "kind": "step",
        "status": "SUCCEEDED",
        "raw_status": "COMPLETED",
        "external_id": "123.0",
        "name": "my-job",
        "command": ["python", "-m", "x"],
        "resources": {"cpus_per_task": 4},
        "log_paths": {"stdout": "/tmp/out.log"},
        "error": None,
        "creation_dt": "2026-02-10T10:00:00",
        "update_dt": "2026-02-10T10:07:00",
    }
    get_job = mocker.patch.object(mod.SwarmStateManager, "get_job", return_value=row)
    refreshed = dict(row)
    refreshed["status"] = "RUNNING"
    refreshed["raw_status"] = "RUNNING"
    refreshed["refresh_source"] = "slurm"
    refreshed["refresh_error"] = None
    swarm = mocker.MagicMock()
    swarm.refresh_job_status.return_value = refreshed
    from_state = mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)

    result = runner.invoke(mod.job_app, ["status", "job-1", "--refresh", "--json"])

    assert result.exit_code == 0
    get_job.assert_called_once_with("job-1")
    from_state.assert_called_once_with(deployment_name="my-swarm")
    swarm.refresh_job_status.assert_called_once_with("job-1")
    payload = _parse_last_json_line(result.stdout)
    assert payload["command"] == "status"
    assert payload["swarm"] == "my-swarm"
    assert payload["job"]["job_id"] == "job-1"
    assert payload["job"]["status"] == "RUNNING"
    assert payload["job"]["refresh_source"] == "slurm"


def test_status_job_tui_uses_renderer(mocker):
    row = {
        "job_id": "job-1",
        "deployment_name": "my-swarm",
        "provider": "slurm",
        "kind": "step",
        "status": "FAILED",
        "raw_status": "FAILED",
        "external_id": "123.0",
        "name": "my-job",
        "command": ["python", "-m", "x"],
        "resources": None,
        "log_paths": None,
        "error": "boom",
        "creation_dt": "2026-02-10T10:00:00",
        "update_dt": "2026-02-10T10:08:00",
    }
    mocker.patch.object(mod.SwarmStateManager, "get_job", return_value=row)
    refreshed = dict(row)
    refreshed["refresh_source"] = "slurm"
    refreshed["refresh_error"] = None
    swarm = mocker.MagicMock()
    swarm.refresh_job_status.return_value = refreshed
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", return_value=swarm)
    render = mocker.patch("domyn_swarm.cli.tui.job_view.render_job_status")

    result = runner.invoke(mod.job_app, ["status", "job-1", "--refresh"])

    assert result.exit_code == 0
    render.assert_called_once()


def test_status_job_refresh_failure_is_best_effort(mocker):
    row = {
        "job_id": "job-1",
        "deployment_name": "my-swarm",
        "provider": "slurm",
        "kind": "step",
        "status": "RUNNING",
        "raw_status": "RUNNING",
        "external_id": "123.0",
        "name": "my-job",
        "command": ["python", "-m", "x"],
        "resources": None,
        "log_paths": None,
        "error": None,
        "creation_dt": "2026-02-10T10:00:00",
        "update_dt": "2026-02-10T10:08:00",
    }
    mocker.patch.object(mod.SwarmStateManager, "get_job", return_value=row)
    mocker.patch.object(mod.DomynLLMSwarm, "from_state", side_effect=RuntimeError("down"))

    result = runner.invoke(mod.job_app, ["status", "job-1", "--refresh", "--json"])

    assert result.exit_code == 0
    payload = _parse_last_json_line(result.stdout)
    assert payload["job"]["job_id"] == "job-1"
    assert payload["job"]["refresh_source"] == "backend"
    assert "down" in payload["job"]["refresh_error"]


def test_status_job_without_refresh_skips_backend(mocker):
    row = {
        "job_id": "job-1",
        "deployment_name": "my-swarm",
        "provider": "slurm",
        "kind": "step",
        "status": "SUCCEEDED",
        "raw_status": "COMPLETED",
        "external_id": "123.0",
        "name": "my-job",
        "command": ["python", "-m", "x"],
        "resources": None,
        "log_paths": None,
        "error": None,
        "creation_dt": "2026-02-10T10:00:00",
        "update_dt": "2026-02-10T10:07:00",
    }
    get_job = mocker.patch.object(mod.SwarmStateManager, "get_job", return_value=row)
    from_state = mocker.patch.object(mod.DomynLLMSwarm, "from_state")

    result = runner.invoke(mod.job_app, ["status", "job-1", "--json"])

    assert result.exit_code == 0
    get_job.assert_called_once_with("job-1")
    from_state.assert_not_called()
    payload = _parse_last_json_line(result.stdout)
    assert payload["job"]["status"] == "SUCCEEDED"
    assert payload["job"]["refresh_source"] == "db"
    assert payload["job"]["refresh_error"] is None


def test_submit_job_accepts_deprecated_num_threads_alias(mocker, tmp_path: Path):
    """`--num-threads` keeps working for one release and says what to use instead."""
    in_path, out_path = _mk_files(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("model: my-model\nname: my-swarm")

    mocker.patch.object(mod, "_load_swarm_config", return_value=object())

    swarm = mocker.MagicMock()
    swarm.endpoint = "http://host:9000"
    swarm.model = "m"
    cm = mocker.MagicMock()
    cm.__enter__.return_value = swarm
    cm.__exit__.return_value = None
    mocker.patch.object(mod, "DomynLLMSwarm", return_value=cm)
    mocker.patch.object(mod.helpers.JobBuilder, "from_class_path", return_value=object())

    res = runner.invoke(
        mod.job_app,
        [
            "submit",
            "--input",
            str(in_path),
            "--output",
            str(out_path),
            "-c",
            str(config_path),
            "--num-threads",
            "5",
        ],
    )

    assert res.exit_code == 0
    assert swarm.submit_job.call_args.kwargs["run"].num_shards == 5
    assert "--num-shards" in res.stderr
