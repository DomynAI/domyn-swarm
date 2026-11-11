# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest

import domyn_swarm.backends.serving.slurm as mod
from domyn_swarm.backends.serving.slurm import SlurmServingBackend
from domyn_swarm.platform.protocols import ServingHandle, ServingPhase


class FakeDriver:
    def __init__(self, submit_jobid=111, submit_lb_jobid=222):
        self.submit_jobid = submit_jobid
        self.submit_lb_jobid = submit_lb_jobid
        self.calls = {"submit_replicas": None, "submit_endpoint": None, "stage": None}
        self.order = []

    def submit_replicas(
        self,
        name,
        replicas,
        nodes,
        gpus_per_node,
        gpus_per_replica,
        replicas_per_node,
        swarm_directory,
    ):
        self.order.append("submit_replicas")
        self.calls["submit_replicas"] = {
            "name": name,
            "replicas": replicas,
            "nodes": nodes,
            "gpus_per_node": gpus_per_node,
            "gpus_per_replica": gpus_per_replica,
            "replicas_per_node": replicas_per_node,
            "swarm_directory": swarm_directory,
        }
        return self.submit_jobid

    def submit_endpoint(self, name, jobid, replicas, swarm_directory):
        self.calls["submit_endpoint"] = {
            "name": name,
            "jobid": jobid,
            "replicas": replicas,
            "swarm_directory": swarm_directory,
        }
        return self.submit_lb_jobid

    def stage_colocated_endpoint(self, name, replicas, swarm_directory):
        self.order.append("stage")
        self.calls["stage"] = {"name": name, "replicas": replicas}


def mk_cfg(port=9000, poll=10, swarm_directory=None, colocate=False):
    if swarm_directory is None:
        swarm_directory = "/home/user/swarms/my-swarm"
    endpoint = SimpleNamespace(port=port, poll_interval=poll, colocate=colocate)
    return SimpleNamespace(endpoint=endpoint, swarm_directory=swarm_directory)


def serving_handle(jobid=101, lb_jobid=202, lb_node=None, url=""):
    meta = {"jobid": jobid, "lb_jobid": lb_jobid}
    if lb_node:
        meta["lb_node"] = lb_node
    return ServingHandle(id=str(lb_jobid), url=url, meta=meta)


# -------------------
# Tests
# -------------------


def test_create_or_update_calls_driver_and_returns_handle(tmp_path):
    driver = FakeDriver(submit_jobid=101, submit_lb_jobid=202)
    cfg = mk_cfg(port=9100, poll=7, swarm_directory=tmp_path)
    be = SlurmServingBackend(driver=driver, cfg=cfg)

    spec = {
        "replicas": 2,
        "nodes": 3,
        "gpus_per_node": 4,
        "gpus_per_replica": 2,
        "replicas_per_node": 2,
        "swarm_directory": tmp_path,
    }

    handle = be.create_or_update("my-swarm", spec, None)

    # Driver calls captured
    assert driver.calls["submit_replicas"] == {
        "name": "my-swarm",
        "replicas": 2,
        "nodes": 3,
        "gpus_per_node": 4,
        "gpus_per_replica": 2,
        "replicas_per_node": 2,
        "swarm_directory": tmp_path,
    }
    assert driver.calls["submit_endpoint"] == {
        "name": "my-swarm",
        "jobid": 101,
        "replicas": 2,
        "swarm_directory": tmp_path,
    }

    # Handle returned
    assert isinstance(handle, ServingHandle)
    assert handle.id == "202"  # lb_jobid as string
    assert handle.url == ""
    assert handle.meta["jobid"] == 101
    assert handle.meta["lb_jobid"] == 202
    assert handle.meta["port"] == 9100
    assert handle.meta["name"] == "my-swarm"


def test_wait_ready_uses_injected_readiness(monkeypatch):
    class FakeProbe:
        def __init__(self):
            self.calls = []

        def wait_ready(self, handle, timeout_s):
            self.calls.append((handle, timeout_s))
            handle.url = "http://node:9000"
            handle.meta["lb_node"] = "node"
            return handle

    driver = FakeDriver()
    cfg = mk_cfg(port=9000, poll=5)
    probe = FakeProbe()
    be = SlurmServingBackend(driver=driver, cfg=cfg, readiness=probe)

    handle = ServingHandle(id="202", url="", meta={"jobid": 101, "lb_jobid": 202})
    out = be.wait_ready(handle, 30, {})

    assert out.url == "http://node:9000"
    assert probe.calls == [(handle, 30)]  # used injected probe


def test_wait_ready_constructs_probe_with_cfg_values(monkeypatch):
    constructed = {}

    class FakeProbeCtor:
        def __init__(
            self, driver, endpoint_port, poll_interval_s, watchdog_db, swarm_name, endpoint_job_file
        ):
            constructed["driver"] = driver
            constructed["endpoint_port"] = endpoint_port
            constructed["poll_interval_s"] = poll_interval_s
            constructed["watchdog_db"] = watchdog_db
            constructed["swarm_name"] = swarm_name
            constructed["endpoint_job_file"] = endpoint_job_file

        def wait_ready(self, handle, timeout_s):
            handle.url = f"http://lb:{constructed['endpoint_port']}"
            handle.meta["lb_node"] = "lb"
            return handle

    # Patch the symbol used by the module
    monkeypatch.setattr(mod, "SlurmReadiness", FakeProbeCtor)

    driver = FakeDriver()
    cfg = mk_cfg(port=8123, poll=1.5)
    be = SlurmServingBackend(driver=driver, cfg=cfg)  # no readiness injected

    handle = ServingHandle(
        id="202", url="", meta={"jobid": 101, "lb_jobid": 202, "name": "my-swarm"}
    )
    out = be.wait_ready(handle, 5, {"swarm_directory": "swarm-directory"})

    assert constructed["driver"] is driver
    assert constructed["endpoint_port"] == 8123
    assert constructed["poll_interval_s"] == 1.5
    assert constructed["watchdog_db"] == Path("swarm-directory/watchdog.db")
    assert constructed["swarm_name"] == "my-swarm"
    assert constructed["endpoint_job_file"] is None
    # watchdog_db should be derived from default swarm_directory in cfg
    assert out.url == "http://lb:8123"
    assert out.meta["lb_node"] == "lb"


def test_delete_scancels_both(monkeypatch):
    calls = []

    def fake_cancel(job_id):
        calls.append(("scancel", str(job_id)))

    monkeypatch.setattr(mod, "_cancel_slurm", fake_cancel)
    driver = FakeDriver()
    cfg = mk_cfg()
    be = SlurmServingBackend(driver=driver, cfg=cfg)

    h = ServingHandle(id="202", url="", meta={"jobid": 101, "lb_jobid": 202})
    be.delete(h)

    assert calls == [("scancel", "101"), ("scancel", "202")]


def test_delete_ignores_missing_ids(monkeypatch):
    calls = []

    def fake_cancel(job_id):
        calls.append(("scancel", str(job_id)))

    monkeypatch.setattr(mod, "_cancel_slurm", fake_cancel)
    be = SlurmServingBackend(driver=FakeDriver(), cfg=mk_cfg())

    # Only lb_jobid present
    h = ServingHandle(id="x", url="", meta={"lb_jobid": 555})
    be.delete(h)

    assert calls == [("scancel", "555")]


def test_ensure_ready_raises_when_handle_none():
    be = SlurmServingBackend(driver=FakeDriver(), cfg=mk_cfg())
    with pytest.raises(RuntimeError, match="Serving handle is null"):
        be.ensure_ready(None)


@pytest.mark.parametrize(
    "meta,url_ok",
    [
        ({"jobid": 1, "lb_jobid": 2, "lb_node": None}, True),
        ({"jobid": 1, "lb_jobid": None, "lb_node": "n"}, True),
        ({"jobid": None, "lb_jobid": 2, "lb_node": "n"}, True),
        ({}, True),
    ],
)
def test_ensure_ready_raises_when_missing_fields(meta, url_ok):
    be = SlurmServingBackend(driver=FakeDriver(), cfg=mk_cfg())
    handle = ServingHandle(id="h", url=("ok" if url_ok else ""), meta=meta)
    with pytest.raises(RuntimeError, match="Swarm not ready"):
        be.ensure_ready(handle)


def test_ensure_ready_ok():
    be = SlurmServingBackend(driver=FakeDriver(), cfg=mk_cfg())
    handle = ServingHandle(
        id="h",
        url="http://n:9000",
        meta={"jobid": 1, "lb_jobid": 2, "lb_node": "node"},
    )
    # Should not raise
    be.ensure_ready(handle)


def test_status_failed_on_bad_state(mocker):
    # rep in BAD → FAILED
    driver = mocker.Mock()
    driver.get_job_state.side_effect = lambda jid: "FAILED" if jid == 101 else "RUNNING"
    driver.get_node_from_jobid.return_value = "n1"

    # requests shouldn't even matter here; but patch anyway
    mocker.patch.object(mod, "requests", SimpleNamespace(get=mocker.Mock()))

    be = SlurmServingBackend(driver=driver, cfg=mk_cfg())

    h = serving_handle()
    st = be.status(h)

    assert st.phase == ServingPhase.FAILED
    # job states checked
    assert driver.get_job_state.call_count == 2


def test_status_pending_when_waiting_or_unknown(mocker):
    driver = mocker.Mock()
    # replica PENDING, lb RUNNING → PENDING
    driver.get_job_state.side_effect = lambda jid: "PENDING" if jid == 101 else "RUNNING"
    driver.get_node_from_jobid.return_value = "n1"
    mocker.patch.object(mod, "requests", SimpleNamespace(get=mocker.Mock()))

    be = SlurmServingBackend(driver=driver, cfg=mk_cfg())

    h = serving_handle()
    st = be.status(h)
    assert st.phase == ServingPhase.PENDING


def test_status_initializing_when_http_not_ready(mocker):
    from types import SimpleNamespace

    import requests

    # Slurm says both jobs are RUNNING
    driver = mocker.Mock()
    driver.get_job_state.side_effect = lambda jid: "RUNNING"
    driver.get_node_from_jobid.return_value = "nX"

    # Patch the module's requests.get to raise a real RequestException
    get_mock = mocker.Mock(side_effect=requests.RequestException("boom"))
    mocker.patch.object(mod, "requests", SimpleNamespace(get=get_mock))

    be = SlurmServingBackend(driver=driver, cfg=mk_cfg(port=8123))

    # No lb_node in handle → backend must resolve node via driver
    h = serving_handle(lb_node=None)  # lb_jobid should be 202 per your helper
    st = be.status(h)

    # Node was looked up once
    driver.get_node_from_jobid.assert_called_once_with(202)

    # We attempted exactly one HTTP probe to /health with a small timeout
    get_mock.assert_called_once()
    url_arg = get_mock.call_args.args[0]
    assert url_arg == "http://nX:8123/health"
    assert get_mock.call_args.kwargs.get("timeout") in (1.0, 1.5, 2.0)

    # Since HTTP isn't ready yet, phase is INITIALIZING and URL is unchanged
    assert st.phase == ServingPhase.INITIALIZING
    assert h.url in ("", None)


def test_status_running_sets_url_and_uses_existing_lb_node(mocker):
    driver = mocker.Mock()
    driver.get_job_state.side_effect = lambda jid: "RUNNING"
    driver.get_node_from_jobid.return_value = "should_not_be_called"

    # HTTP 200 OK
    http_ok = SimpleNamespace(status_code=200)
    req = mocker.Mock()
    req.get.return_value = http_ok
    mocker.patch.object(mod, "requests", req)

    be = SlurmServingBackend(driver=driver, cfg=mk_cfg(port=9001))

    # Provide lb_node so driver.get_node_from_jobid is NOT called
    h = serving_handle(lb_node="nZZ")
    st = be.status(h)

    driver.get_node_from_jobid.assert_not_called()
    assert st.phase == ServingPhase.RUNNING
    assert h.url == "http://nZZ:9001"
    req.get.assert_called_once()
    assert req.get.call_args.kwargs.get("timeout") in (1.5, 1, 2)  # small timeout


def test_colocated_endpoint_is_staged_before_replicas_and_not_submitted(tmp_path):
    driver = FakeDriver(submit_jobid=101)
    be = SlurmServingBackend(driver=driver, cfg=mk_cfg(colocate=True))

    handle = be.create_or_update("my-swarm", {"replicas": 2, "swarm_directory": tmp_path}, {})

    assert driver.order == ["stage", "submit_replicas"]
    assert driver.calls["submit_endpoint"] is None
    assert handle.meta["jobid"] == handle.meta["lb_jobid"] == 101


def test_colocated_wait_ready_reads_the_endpoint_job_file(monkeypatch, tmp_path):
    constructed = {}

    class FakeProbeCtor:
        def __init__(self, **kwargs):
            constructed.update(kwargs)

        def wait_ready(self, handle, timeout_s):
            return handle

    monkeypatch.setattr(mod, "SlurmReadiness", FakeProbeCtor)
    be = SlurmServingBackend(driver=FakeDriver(), cfg=mk_cfg(colocate=True))
    handle = ServingHandle(
        id="101",
        url="",
        meta={"jobid": 101, "lb_jobid": 101, "name": "s", "swarm_directory": str(tmp_path)},
    )

    be.wait_ready(handle, 5, {})

    assert constructed["endpoint_job_file"] == tmp_path / "serving" / "endpoint.job"


def test_delete_cancels_a_colocated_swarm_once(monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "_cancel_slurm", calls.append)
    be = SlurmServingBackend(driver=FakeDriver(), cfg=mk_cfg(colocate=True))

    be.delete(serving_handle(jobid=101, lb_jobid=101))

    assert calls == ["101"]
