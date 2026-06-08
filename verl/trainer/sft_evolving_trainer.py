# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Self-evolving SFT-distillation trainer.

A fork of `RayPPOTrainer` whose per-step *learning* is supervised fine-tuning on
teacher reasoning traces (cross-entropy via `sft_loss`) instead of policy
gradient — this is NOT on-policy distillation (no teacher logprobs, no PPO).

What we reuse from the RL pipeline (the expensive, battle-tested parts):
  * `ActorRolloutRefWorker` hybrid engine (train + colocated vLLM + weight sync),
  * `checkpoint_manager.update_weights()` for step-wise FSDP→vLLM weight sync,
  * `_validate()` for generation-based validation logged to wandb,
  * `RewardLoopManager` + the self_evolving reward fn, which already POSTs
    per-question accuracy to the gen server's `/report` for difficulty feedback,
  * `_update_actor()` — it converts the padded batch to no-padding and calls the
    actor engine, so swapping the loss to `sft_loss` makes it an SFT step.

What changes vs. RL:
  * the training gradient is teacher-forced on full traces fetched from the gen
    server (SelfEvolvingSFTDataset), so there is NO rollout / advantage / critic
    / reference policy in the train step;
  * vLLM is only woken for evaluation (validation + generated-question feedback),
    and put back to sleep for training — generation is not on the gradient path.
"""

from functools import partial
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import extract_reward
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking
from verl.workers.utils.losses import sft_loss


def _pad_and_generate(trainer, gen_batch):
    """Pad a gen batch to the rollout worker count, generate, unpad."""
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

    size_divisor = trainer.config.actor_rollout_ref.rollout.agent.num_workers
    padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
    out = trainer.async_rollout_manager.generate_sequences(padded)
    return unpad_dataproto(out, pad_size=pad_size)


class SelfEvolvingSFTTrainer(RayPPOTrainer):
    """RL-infra trainer with an SFT-distillation learning step."""

    # ------------------------------------------------------------------
    # Worker / loss setup
    # ------------------------------------------------------------------
    def init_workers(self):
        super().init_workers()
        # Override the actor's policy-gradient loss with plain SFT cross-entropy.
        # `sft_loss` ignores its `config` arg; it reads loss_mask / pad_mode from
        # the per-batch tensordict, both of which `_update_actor` provides after
        # `left_right_2_no_padding` (which sets loss_mask = response_mask).
        self.actor_rollout_wg.set_loss_fn(partial(sft_loss, config=None))
        print("SelfEvolvingSFTTrainer: actor loss set to sft_loss")

    # ------------------------------------------------------------------
    # Dataloaders: SFT train (full traces) + RL val (prompt-only) + feedback
    # ------------------------------------------------------------------
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        from torchdata.stateful_dataloader import StatefulDataLoader

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn

        if collate_fn is None:
            collate_fn = rl_collate_fn
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, train_dataset)

        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        num_workers = self.config.data["dataloader_num_workers"]

        # SFT train: each sample is already tokenized (input_ids, attention_mask,
        # position_ids, loss_mask) by SelfEvolvingSFTDataset; rl_collate stacks them.
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        # Validation: fixed prompt-only set for generation → wandb metrics.
        val_batch_size = self.config.data.val_batch_size or len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Feedback: prompt-only gen-server questions the student answers so the
        # reward fn can POST difficulty back to the gen server. Optional. We load
        # the prompt-only SelfEvolvingDataset directly (it reads the same
        # `config.data.self_evolving.gen_server_url`) rather than via custom_cls,
        # which is already bound to the SFT trace dataset.
        self.feedback_dataloader = None
        fb_cfg = self.config.data.get("feedback", {})
        if fb_cfg and fb_cfg.get("enable", False):
            from verl.utils.import_utils import load_extern_type

            cls = load_extern_type(fb_cfg.custom_cls.path, fb_cfg.custom_cls.name)
            feedback_dataset = cls(
                data_files=self.config.data.train_files,
                tokenizer=self.tokenizer,
                config=self.config.data,
                processor=self.processor,
                max_samples=-1,
            )
            self.feedback_dataloader = StatefulDataLoader(
                dataset=feedback_dataset,
                batch_size=fb_cfg.get("batch_size", self.config.data.val_batch_size or 64),
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
            )
            self._feedback_iter = iter(self.feedback_dataloader)

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        print(
            f"SFT train dataloader: {len(self.train_dataloader)}, "
            f"val dataloader: {len(self.val_dataloader)}, "
            f"feedback: {'on' if self.feedback_dataloader else 'off'}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        from omegaconf import OmegaConf, open_dict

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: could not set total_training_steps in config: {e}")

    # ------------------------------------------------------------------
    # Generated-question feedback: generate → reward (auto /report) → log
    # ------------------------------------------------------------------
    def _feedback_eval(self) -> dict:
        if self.feedback_dataloader is None:
            return {}
        try:
            batch_dict = next(self._feedback_iter)
        except StopIteration:
            self._feedback_iter = iter(self.feedback_dataloader)
            batch_dict = next(self._feedback_iter)

        import uuid

        batch = DataProto.from_single_dict(batch_dict)
        if "uid" not in batch.non_tensor_batch:
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )

        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
            "global_steps": self.global_steps,
        }
        out = _pad_and_generate(self, gen_batch)

        # Reward (and the self_evolving reward fn's /report side effect) is
        # computed by the streaming agent reward loop during generation when
        # there is no reward model; otherwise compute it on the colocated RM.
        if self.use_rm and "rm_scores" not in out.batch.keys():
            self.checkpoint_manager.sleep_replicas()
            out = out.union(self._compute_reward_colocate(out))
            self.checkpoint_manager.update_weights(self.global_steps)

        batch = batch.union(out)
        try:
            reward_tensor, _ = extract_reward(batch)
            scores = reward_tensor.sum(-1).cpu().tolist()
        except Exception as e:  # be robust: feedback must never crash training
            print(f"feedback reward extraction failed: {e}")
            return {}
        if not scores:
            return {}
        return {
            "feedback/reward_mean": float(np.mean(scores)),
            "feedback/reward_min": float(np.min(scores)),
            "feedback/reward_max": float(np.max(scores)),
            "feedback/n": float(len(scores)),
        }

    # ------------------------------------------------------------------
    # SFT training step
    # ------------------------------------------------------------------
    def _sft_train_step(self, batch_dict) -> dict:
        batch = DataProto.from_single_dict(batch_dict)
        # left_right_2_no_padding (inside _update_actor) requires "response_mask"
        # and sets the nested loss_mask = response_mask, which sft_loss reads.
        if "response_mask" not in batch.batch.keys():
            assert "loss_mask" in batch.batch.keys(), "SFT batch must carry loss_mask"
            batch.batch["response_mask"] = batch.batch["loss_mask"]
        actor_output = self._update_actor(batch)
        metrics = reduce_metrics(actor_output.meta_info["metrics"])
        if "actor/loss" in metrics:
            metrics["train/loss"] = metrics["actor/loss"]
        return metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(self):
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.config.trainer.get("val_before_train", True):
            self.checkpoint_manager.update_weights(self.global_steps)
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            self.checkpoint_manager.sleep_replicas()
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="SFT Training")
        self.global_steps += 1
        last_val_metrics = None
        current_epoch = self.global_steps // len(self.train_dataloader)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                is_last_step = self.global_steps >= self.total_training_steps

                # --- SFT learning step (vLLM asleep; teacher-forced) ---------
                metrics.update(self._sft_train_step(batch_dict))

                is_eval_step = self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                )
                # --- evaluation: sync weights → generate → reward/wandb ------
                if is_eval_step:
                    self.checkpoint_manager.update_weights(self.global_steps)
                    val_metrics = self._validate()
                    metrics.update(val_metrics)
                    if is_last_step:
                        last_val_metrics = val_metrics
                    metrics.update(self._feedback_eval())
                    self.checkpoint_manager.sleep_replicas()

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()

                metrics["training/global_step"] = self.global_steps
                metrics["training/epoch"] = epoch
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
