def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
    """
    Creates the train and validation dataloaders.
    """
    # TODO: we have to make sure the batch size is divisible by the dp size
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

    if train_dataset is None:
        train_dataset = create_rl_dataset(
            self.config.data.train_files, self.config.data, self.tokenizer, self.processor
        )
    if val_dataset is None:
        val_dataset = create_rl_dataset(
            self.config.data.val_files, self.config.data, self.tokenizer, self.processor
        )
    self.train_dataset, self.val_dataset = train_dataset, val_dataset

    if train_sampler is None:
        # TODO; you can essentially specify the type of sampler here, based also on the data split
        train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
    if collate_fn is None:
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

        collate_fn = default_collate_fn

    num_workers = self.config.data["dataloader_num_workers"]

    ## TODO: trainer_sampler is pretty much the rl_sampler here, which shuffles the dataset
    # TODO: the sampler is now placed into this (which is essentially your sampler)
    self.train_dataloader = StatefulDataLoader(
        dataset=self.train_dataset,
        batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
        num_workers=num_workers,
        # shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )

    val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
    if val_batch_size is None:
        val_batch_size = len(self.val_dataset)

    # TODO: validation data is shuffled here as well
    self.val_dataloader = StatefulDataLoader(
        dataset=self.val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=self.config.data.get("validation_shuffle", True),
        drop_last=False,
        collate_fn=collate_fn,
    )

    assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
    assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

    print(
        f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
        f"{len(self.val_dataloader)}"
    )

    total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

    if self.config.trainer.total_training_steps is not None:
        total_training_steps = self.config.trainer.total_training_steps

    self.total_training_steps = total_training_steps
    print(f"Total training steps: {self.total_training_steps}")

    try:
        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            if OmegaConf.select(self.config, "critic.optim"):
                self.config.critic.optim.total_training_steps = total_training_steps
    except Exception as e:
        print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")