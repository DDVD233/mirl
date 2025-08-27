import wandb


def init_wandb(project: str, entity: str | None, config: dict, run_name: str | None = None):
    """Initialize a wandb run and log a config table."""
    run = wandb.init(project=project, entity=entity, config=config, name=run_name)
    # Log config as a table for readability
    config_table = wandb.Table(columns=["Parameter", "Value"])
    for key, value in config.items():
        config_table.add_data(key, str(value))
    wandb.log({"configuration": config_table})
    return run


def log_metrics(split_name: str, metrics: dict, epoch: int | None = None):
    """Log a flat dict of metrics with split prefix. Accepts already-prefixed keys too."""
    if not wandb.run:
        return
    log_dict = {}
    if epoch is not None:
        log_dict["epoch"] = epoch
    for key, value in metrics.items():
        # If key already has split prefix, keep it, else prefix
        if key.startswith(f"{split_name}/"):
            log_dict[key] = value
        else:
            log_dict[f"{split_name}/{key}"] = value
    wandb.log(log_dict)


def log_confusion_matrix(split_name: str, y_true: list[int], y_pred: list[int], class_names: list[str]):
    if not wandb.run:
        return
    wandb.log({
        f"{split_name}/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=class_names,
        )
    })


def log_line_series(name: str, xs: list[int], ys_series: list[list[float]], keys: list[str], title: str, xname: str = "Epoch"):
    if not wandb.run:
        return
    wandb.log({
        name: wandb.plot.line_series(
            xs=xs,
            ys=ys_series,
            keys=keys,
            title=title,
            xname=xname,
        )
    })


def finish():
    if wandb.run:
        wandb.finish()


