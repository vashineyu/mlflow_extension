import pytest
import pathlib
import numpy as np

import mlflow
from _pytest.config import Config
from ..utils import (
    Teleport,
    set_experiment
)
from .. import mlflow_wrapper
from ..mlflow_func import log_params, log_metrics, log_tags


@pytest.fixture(scope='session')
def mlruns_dir(pytestconfig: Config):
    return pathlib.Path(pytestconfig.cache.makedir('mlruns_dir'))


@pytest.fixture
def teleport(mlruns_dir):
    return Teleport(
        experiment_name="it-is-unittest1",
        tracking_url=str(mlruns_dir),
        artifact_location=str(mlruns_dir),
    )


@pytest.fixture
def tags_group1():
    return {
        "tag-group1-1": 123,
        "tag-group1-2": 456,
    }


@pytest.fixture
def tags_group2():
    return {
        "tag-group2-1": "foo",
        "tag-group2-2": "bar",
    }


@pytest.fixture
def tags_group3():
    return {
        "tag-group3-1": "hi",
        "tag-group3-2": 666,
    }


@pytest.fixture
def params_group1():
    return {
        "params-group1-1": "a-param-string",
        "params-group1-2": 777,
    }


class PseudoTrainer:
    @mlflow_wrapper.param()
    def __init__(self, params1, params2):
        self.params1 = params1
        self.params2 = params2

    @mlflow_wrapper.metric(names=["log-val1", "log-val2"])
    def train(self, *args, **kwargs):
        return np.random.random(), np.random.random()


@mlflow_wrapper.metric(names=["log-val3", "log-val4"], collect=["log-val4"])
@mlflow_wrapper.param(collect=["params3"])
def pseudofunction(params3, params4, *args, **kwargs):
    return np.random.random(), np.random.random()


def fetch_check_data(teleport):
    exp_id = mlflow.get_experiment_by_name(teleport.experiment_name).experiment_id
    run_info = mlflow.list_run_infos(experiment_id=exp_id)[0]
    run_data = mlflow.get_run(run_info.run_id)
    return run_data


### Start Testing ###
def test_mlflow_with_class(teleport, tags_group1):
    set_experiment(teleport)
    log_tags(tags_group1)
    trainer = PseudoTrainer(
        params1=8,
        params2=9,
    )
    _ = trainer.train()

    run_data = fetch_check_data(teleport)
    fetched_params = run_data.data.params
    fetched_tags = run_data.data.tags
    fetched_metrics = run_data.data.metrics

    assert int(fetched_params["params1"]) == trainer.params1
    assert int(fetched_params["params2"]) == trainer.params2
    assert int(fetched_tags["tag-group1-1"]) == tags_group1["tag-group1-1"]
    assert int(fetched_tags["tag-group1-2"]) == tags_group1["tag-group1-2"]
    assert "log-val1" in fetched_metrics
    assert "log-val2" in fetched_metrics


def test_mlflow_with_function(teleport, tags_group2):
    set_experiment(teleport)
    log_tags(tags_group2)
    _ = pseudofunction(
        params3=5,
        params4=6
    )

    run_data = fetch_check_data(teleport)
    fetched_params = run_data.data.params
    fetched_metrics = run_data.data.metrics
    assert int(fetched_params["params3"]) == 5
    assert "params4" not in fetched_params
    assert "log-val3" not in fetched_metrics
    assert "log-val4" in fetched_metrics


def test_mlflow_standalone(teleport, tags_group3, params_group1):
    set_experiment(teleport)
    log_tags(tags_group3)
    log_params(params_group1)
    log_metrics({"log-val4": np.random.random()})

    run_data = fetch_check_data(teleport)
    fetched_params = run_data.data.params
    fetched_metrics = run_data.data.metrics
    assert fetched_params["params-group1-1"] == "a-param-string"
    assert int(fetched_params["params-group1-2"]) == 777
    assert "log-val4" in fetched_metrics
