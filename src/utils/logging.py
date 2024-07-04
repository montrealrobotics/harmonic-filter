"""
Source: https://github.com/ashleve/lightning-hydra-template/blob/main/train.py
"""
from typing import Sequence

import logging
import warnings
import numpy as np
import os
import random
import pandas as pd
import json

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed_value):
    np.random.seed(seed_value)  # Seed NumPy
    random.seed(seed_value)  # Seed Python's random module


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    return logger


log = get_logger(__name__)


def log_experiment_info(cfg: DictConfig, data_path: str) -> None:
    # Only export comet when we are going to log everything
    import comet_ml
    from comet_ml import Experiment

    # Initialize logger if provided
    if "logger" in cfg:
        log.info(f"Instantiating logger...")
        attr = cfg.logger.comet
        comet_ml.init(project_name=attr.project_name, workspace=attr.workspace)
        experiment = Experiment()
        # Log training args
        experiment.add_tag(cfg.logger.tag)
    else:
        # There is no logger, stop here
        return None

    # Log config
    experiment.log_asset_data(OmegaConf.to_container(cfg), 'config.yaml')
    experiment.log_parameters(OmegaConf.to_container(cfg))

    # Log video (if available)
    video_path = os.path.join(data_path, "result.mp4")
    if os.path.exists(video_path):
        log.info(f"Logging video at path: {video_path}")
        experiment.log_image(video_path, image_format="mp4")
    # Log MAP video (if available)
    video_path = os.path.join(data_path, "map_video.mp4")
    if os.path.exists(video_path):
        log.info(f"Logging MAP video at path: {video_path}")
        experiment.log_image(video_path, image_format="mp4")

    # List al .png files in data_path
    others_path = os.path.join(data_path, "others")
    if os.path.exists(others_path):
        png_files = [f for f in os.listdir(others_path) if f.endswith(".png")]
        # Log images to comet
        if len(png_files) > 0:
            log.info("Logging images...")
            for png_file in png_files:
                experiment.log_image(os.path.join(others_path, png_file))

    # See if there is a pickle with metrics and log those
    metrics_path = os.path.join(others_path, "results.json")
    if os.path.exists(metrics_path):
        log.info(f"Logging metrics at: {metrics_path}")
        # Read the json file
        with open(metrics_path, 'rb') as f:
            metrics = json.load(f)
        experiment.log_asset_data(metrics, 'results.json')
        # Log table and ll
        for key, value in metrics.items():
            if key == "neg_log_likelihood":
                # Log filter by filter in comet
                for filter_name, data in value.items():
                    for it, sample in enumerate(data):
                        experiment.log_metric(value=sample, name=f"{filter_name}/{key}", epoch=it)
            elif key == 'metrics':
                # Save metrics independently as a JSON to log into comet
                table_path = os.path.join(others_path, "table.csv")
                # Create dataframe from table for logging
                df = pd.DataFrame(value).T
                df.insert(0, 'Filter', df.index)
                df.to_csv(table_path, index=False)
                # Finally, log table
                experiment.log_table(table_path)
                # Log one metric from the table to comet such that we can sort easily
                for filter_name, nll in zip(df.index, df["MeanNLL"].to_numpy()):
                    experiment.log_metric(value=nll, name=f"{filter_name}/MeanNLL")
                # Log one metric from the table to comet such that we can sort easily
                for filter_name, rmse in zip(df.index, df["RMSE"].to_numpy()):
                    experiment.log_metric(value=rmse, name=f"{filter_name}/RMSE")
                # Log one metric from the table to comet such that we can sort easily
                for filter_name, rmse_map in zip(df.index, df["RMSE_MAP"].to_numpy()):
                    experiment.log_metric(value=rmse_map, name=f"{filter_name}/RMSE_MAP")

    # Log job config
    config_tree_path = os.path.join(others_path, "config_tree.log")
    if os.path.exists(config_tree_path):
        log.info("Logging config tree")
        cfg_tree = open(config_tree_path, "r")
        experiment.log_asset_data(cfg_tree.read(), 'config_tree.log')
        cfg_tree.close()


def extras(config: DictConfig, logging_path: str) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, logging_path, resolve=True)


def print_config(
        config: DictConfig,
        logging_path: str,
        print_order: Sequence[str] = (
                "logger",
                "launcher",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        logging_path (str): Path to store config tree
        print_order ():
        config (DictConfig): Configuration composed by Hydra.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(f"Field '{field}' not found in config")

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    with open(os.path.join(logging_path, "config_tree.log"), "w") as file:
        rich.print(tree, file=file)
