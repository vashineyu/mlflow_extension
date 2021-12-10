import sys
import warnings
from glob import glob

from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger import LoggerHook
from mmcv.runner.iter_based_runner import IterBasedRunner
from mmcv.utils import Config


if sys.version_info.minor < 8:
    from importlib_metadata import version
else:
    from importlib.metadata import version


try:
    import mlflow
    import mlflow.pytorch as mlflow_pytorch
except ImportError:
    warnings.warn(
        'You have registered MlFlowTrack. '
        'But some required dependencies are not installed: mlflow',
        ImportWarning,
    )


@HOOKS.register_module(name="MlflowLoggerHook", force=True)
class MlFlowTrack(LoggerHook):
    """
    Hooks for mlflow tracking

    Args:
        exp_name (str): Experiment name. Should be set as your project name.
        log_model (bool): record model file as artifact (recommend not to do it, waste space)
        tags (dict): key-value pairs for custom tags (for customized filtering)
        additional_params (dict): additional parameters to record (key-value pairs)
        tracking_uri (str): destination to the tracking-uri (ex. http://0.0.0.0:8050)
        artifact_location (str): path to folder or s3 (s3://[bucket-name])
    """
    def __init__(
        self,
        exp_name='default',
        run_id=None,
        log_model=False,
        tags=None,
        additional_params=None,
        tracking_uri=None,
        artifact_location=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.exp_name = exp_name
        self.run_id = run_id
        self.log_model = log_model
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.tags = tags or {}
        self.additional_params = additional_params or {}

    @master_only
    def before_run(self, runner):
        super().before_run(runner)
        config = self.get_config(runner)

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        if self.artifact_location and mlflow.get_experiment_by_name(self.exp_name) is None:
            # Create a experiment if #artifact_location is specified
            mlflow.create_experiment(self.exp_name, self.artifact_location)

        mlflow.set_experiment(self.exp_name)
        if self.run_id:
            mlflow.start_run(self.run_id)
        mlflow.set_tags(self.get_tags())

        # Automatically set flags via runner type.
        if isinstance(runner, IterBasedRunner):
            self.by_epoch = False

        # Log some hyper-parameters.
        recordings = {
            "model_arch": config["model"]["type"],
            "backbone": config["model"]["backbone"]["type"],
            "samples_per_gpu": config["data"]["samples_per_gpu"],
            "optimizer": config["optimizer"]["type"],
            "init_lr": config["optimizer"]["lr"],
            "num_epochs": runner.max_epochs,
            "num_iters": runner.max_iters,
        }
        recordings.update(self.additional_params)
        mlflow.log_params(recordings)

    @master_only
    def log(self, runner):
        metrics = self.get_loggable_tags(runner)
        if metrics:
            mlflow.log_metrics(metrics, step=self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        if self.log_model:
            mlflow_pytorch.log_model(runner.model, 'models')

        # get saved config path as artifact
        work_dir = self.get_config(runner)["work_dir"]
        config_path = glob("{}/*.py".format(work_dir))[0]

        mlflow.log_artifact(config_path)

    def get_config(self, runner):
        return Config.fromstring(runner.meta["config"], ".py").to_dict()

    def get_tags(self):
        tags = {
            "mmcv-full": version("mmcv-full"),
            "torch": version("torch"),
        }
        tags.update(self.tags)
        return tags
