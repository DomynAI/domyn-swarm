# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import ANY, MagicMock, patch

import jinja2
import pytest

from domyn_swarm.backends.serving import slurm_driver as slurm_driver_mod
from domyn_swarm.backends.serving.slurm_driver import SlurmDriver
from domyn_swarm.config.router import VllmRouter
from domyn_swarm.config.slurm import (
    SlurmConfig,
    SlurmEndpointConfig,
)
from domyn_swarm.config.swarm import DomynLLMSwarmConfig


@pytest.fixture
def dummy_config(tmp_path):
    return DomynLLMSwarmConfig(
        name="gpt4",
        image="/path/to/vllm.sif",
        model="gpt-4",
        replicas=2,
        template_path=tmp_path / "template.sh.j2",
        log_directory=tmp_path / "logs",
        home_directory=tmp_path,
        backend=SlurmConfig(
            type="slurm",
            partition="debug",
            account="test_account",
            qos="test_qos",
            endpoint=SlurmEndpointConfig(
                nginx_image="/path/to/nginx.sif",
            ),
        ).model_dump(),
    )


@pytest.fixture
def slurm_driver(dummy_config):
    return SlurmDriver(cfg=dummy_config)


@patch("domyn_swarm.backends.serving.slurm_driver.subprocess.check_output")
@patch("domyn_swarm.backends.serving.slurm_driver.jinja2.Environment.get_template")
def test_submit_replicas(mock_get_template, mock_check_output, slurm_driver, tmp_path):
    mock_template = MagicMock()
    mock_template.render.return_value = "#!/bin/bash\necho test"
    mock_get_template.return_value = mock_template
    mock_check_output.return_value = "12345;dummy"

    job_id = slurm_driver.submit_replicas("test_job", 4, 2, 4, 2, 2, str(tmp_path))
    assert job_id == 12345
    mock_get_template.assert_called()
    mock_check_output.assert_called()


@patch("domyn_swarm.backends.serving.slurm_driver.subprocess.check_output")
@patch("domyn_swarm.backends.serving.slurm_driver.jinja2.Environment.get_template")
def test_submit_endpoint(mock_get_template, mock_check_output, slurm_driver, tmp_path):
    mock_template = MagicMock()
    mock_template.render.return_value = "#!/bin/bash\necho lb"
    mock_get_template.return_value = mock_template
    mock_check_output.return_value = "67890"

    job_id = slurm_driver.submit_endpoint("test_lb_job", 12345, 4, tmp_path)
    assert job_id == 67890
    mock_get_template.assert_called()
    mock_check_output.assert_called_with(
        [
            "sbatch",
            "--parsable",
            "--dependency",
            "after:12345",
            "--export",
            "ALL,DEP_JOBID=12345",
            ANY,
        ],
        text=True,
    )


@pytest.mark.parametrize(
    ("endpoint_qos", "expected_qos"),
    [(None, "test_qos"), ("endpoint_qos", "endpoint_qos")],
)
def test_lb_template_uses_endpoint_qos_override(dummy_config, endpoint_qos, expected_qos):
    dummy_config.backend.endpoint.qos = endpoint_qos
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(dummy_config.backend.template_path.parent),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    rendered = env.get_template("lb.sh.j2").render(
        cfg=dummy_config,
        job_name="test_lb_job",
        dep_jobid=12345,
        replicas=4,
        swarm_directory="/tmp/swarm",
        collector_script_path="/tmp/collector.py",
    )

    assert f"#SBATCH --qos={expected_qos}" in rendered


def _render_lb(cfg):
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(cfg.backend.template_path.parent),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template("lb.sh.j2").render(
        cfg=cfg,
        job_name="test_lb_job",
        dep_jobid=12345,
        replicas=4,
        swarm_directory="/tmp/swarm",
        collector_script_path="/tmp/collector.py",
        supervisor_script_path="/tmp/supervisor.py",
    )


def test_lb_template_defaults_collector_port_to_9100(dummy_config):
    assert 'COLLECTOR_PORT="${COLLECTOR_PORT:-9100}"' in _render_lb(dummy_config)


def test_lb_template_uses_configured_collector_port(dummy_config):
    dummy_config.backend.endpoint.collector_port = 9200
    assert 'COLLECTOR_PORT="${COLLECTOR_PORT:-9200}"' in _render_lb(dummy_config)


@patch("domyn_swarm.backends.serving.slurm_driver.subprocess.check_output")
def test_get_node_from_jobid(mock_check_output, slurm_driver):
    mock_check_output.side_effect = ["nodespec\n", "node001\nnode002\n"]
    node = slurm_driver.get_node_from_jobid(11111)
    assert node == "node001"


@patch("domyn_swarm.backends.serving.slurm_driver.subprocess.check_output")
def test_submit_endpoint_writes_static_server_conf(mock_check_output, slurm_driver, tmp_path):
    mock_check_output.return_value = "55555"
    swarm_dir = tmp_path / "swarms" / "s1"
    (swarm_dir / "serving").mkdir(parents=True)
    slurm_driver.submit_endpoint("job", 12345, 2, str(swarm_dir))
    server_conf = swarm_dir / "serving" / "10-server.conf"
    assert server_conf.exists()
    assert "proxy_pass http://llm;" in server_conf.read_text()


@patch("domyn_swarm.backends.serving.slurm_driver.subprocess.check_output")
@patch("domyn_swarm.backends.serving.slurm_driver.jinja2.Environment.get_template")
def test_consistent_hash_logs_warning(
    mock_get_template, mock_check_output, dummy_config, tmp_path, caplog, monkeypatch
):
    monkeypatch.setattr(slurm_driver_mod.logger, "propagate", True)
    mock_get_template.return_value.render.return_value = "#!/bin/bash\necho lb"
    mock_check_output.return_value = "67890"
    dummy_config.backend.endpoint.router = VllmRouter(
        binary="vllm-router", policy="consistent_hash"
    )

    with caplog.at_level(logging.WARNING):
        SlurmDriver(cfg=dummy_config).submit_endpoint("job", 12345, 2, str(tmp_path))

    assert (
        "vllm-router consistent_hash sends every request without a session ID to one replica"
        in caplog.text
    )
