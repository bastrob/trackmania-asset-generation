from pathlib import Path

import click

from trackmania.config import load_config
from trackmania.tasks.registry import TASKS

BASE_DIR = Path(__file__).resolve().parent.parent


@click.group()
def cli():
    """Trackmania ML CLI"""
    pass

@cli.command()
@click.option("--task", required=True, help="Task to run (base, rough, etc.)")
@click.option("--config", required=False, default=None, help="Configuration to use")
def train(task: str, config: str):
    """
    Run a training task.
    """
    if task not in TASKS:
        raise click.ClickException(f"Unknown task: {task}")

    task_class = TASKS[task]

    config_dict = {}
    if config:
        config_path = BASE_DIR / config
        config_dict = load_config(str(config_path)) if config else {}

    task_instance = task_class(config=config_dict)
    task_instance.run()


if __name__ == "__main__":
    cli()