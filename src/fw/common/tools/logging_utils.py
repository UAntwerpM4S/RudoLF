"""
Shared generic logging utilities used across the framework.
"""

from __future__ import annotations

import os
import csv


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
        print("Warning: No training metrics to write.")
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
