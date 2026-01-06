import wandb

run = wandb.init(project="hier_advantage_distribution_upload")

artifact = wandb.Artifact(name="hier_advantage_distribution", type="dataset")  # type can be "dataset", "model", "backup", etc.
artifact.add_file("/orcd/home/002/keaneong/orcd/pool/human_behaviour_data/hier_advantages.zip")

run.log_artifact(artifact)
run.finish()