import click

from trackmania.tasks.registry import TASKS


@click.group()
def cli():
    """Trackmania ML CLI"""
    pass

@cli.command()
@click.option("--task", required=True, help="Task to run (base, rough, etc.)")
def train(task: str):
    """
    Run a training task.
    """
    if task not in TASKS:
        raise click.ClickException(f"Unknown task: {task}")

    task_class = TASKS[task]
    task_instance = task_class()
    task_instance.run()


if __name__ == "__main__":
    cli()