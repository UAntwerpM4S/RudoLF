"""
Shared generic logging utilities used across the framework.
"""

from __future__ import annotations

import os
import csv
import logging

from pathlib import Path

logger = logging.getLogger(__name__)


def dump_metrics_to_csv(model_dir, csv_filename, training_metrics):
    """
    Write training metrics (loss, advantage, reward, ...) to a CSV file with headers.

    Args:
        model_dir (str): Folder to write the CSV into.
        csv_filename (str): Name of the CSV file.
        training_metrics (list): List of metric dicts; the first dict's keys form the header.
    """

    if not training_metrics:
        logger.warning("No training metrics to write.")
        return

    filepath = os.path.join(model_dir, csv_filename)

    with open(filepath, mode="w", newline="") as file:
        # Assumes all dicts share the same keys.
        fieldnames = training_metrics[0].keys()

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_metrics)


def dump_multi_agent_metrics(model_dir, metrics_dict: dict):
    """
    Flatten per-agent metrics (keys 'agent<N>') into one CSV with an 'agent_id' column.
    """

    flat_metrics = []
    for agent_name, agent_metrics in metrics_dict.items():
        agent_id = int(agent_name.replace('agent', ''))
        for metric in agent_metrics:
            flat_metrics.append({
                'agent_id': agent_id,
                **metric
            })

    dump_metrics_to_csv(model_dir, "training_metrics.csv", flat_metrics)


def create_logfile_handler(_log_file_path: Path, log_level: int):
    """
    Create a UTF-8 logging file handler, creating the parent directory if needed.

    Args:
        _log_file_path: Path to the log file.
        log_level: Minimum logging level for messages written to the file.

    Returns:
        A configured `logging.FileHandler` instance.
    """

    log_dir = _log_file_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    _file_handler = logging.FileHandler(_log_file_path, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    _file_handler.setLevel(log_level)

    return _file_handler
