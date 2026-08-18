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
    Save training metrics to a CSV file.

    This method writes the training metrics (such as loss, advantage, reward)
    to a CSV file with headers.

    Args:
        model_dir (): The folder under which all models are stored.
        csv_filename (str): The name of the CSV file to write.
        training_metrics (list): A list of dictionaries containing the training metrics.

    """

    if not training_metrics:
        logger.warning("No training metrics to write.")
        return

    filepath = os.path.join(model_dir, csv_filename)

    with open(filepath, mode="w", newline="") as file:
        # Get all possible keys (assuming all dicts have the same structure)
        fieldnames = training_metrics[0].keys()

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_metrics)


def dump_multi_agent_metrics(model_dir, metrics_dict: dict):
    """
    Helper method to flatten multi-agent metrics for CSV export.
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
    Create and configure a file handler for logging.

    The parent directory of the specified log file is created if it does not
    already exist. The handler uses UTF-8 encoding and a standard log format
    containing the timestamp, log level, logger name, and message.

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
