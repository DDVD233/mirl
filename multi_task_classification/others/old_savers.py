
    def _create_checkpoint_data(self, optimizer, epoch, scheduler=None, scaler=None):
        
   
        training_strategy = self.global_config.get('TRAINING_STRATEGY', 'head_only')

        

        unwrapped = self.accelerator.unwrap_model(self.model)

        with open("/home/keaneong/human-behavior/verl/multi_task_classification/debug_save.txt", "a") as f:
                        f.write(f"\nStop here, unwrap model")
        model_sd = self.accelerator.get_state_dict(self.model)  # safe across wrappers
        
        with open("/home/keaneong/human-behavior/verl/multi_task_classification/debug_save.txt", "a") as f:
                        f.write(f"Stop here, model_sd")

        classifier_sd = None
        lora_sd = None
        adapter_path = None
        full_model_sd = None

        if training_strategy == "head_only":
            if hasattr(unwrapped, "classifier"):
                classifier_sd = unwrapped.classifier.state_dict()
            elif hasattr(unwrapped, "head"):
                classifier_sd = unwrapped.head.state_dict()
            else:
                classifier_sd = {k: v for k, v in model_sd.items()
                                if k.startswith("classifier.") or k.startswith("head.")}

        elif training_strategy == "lora":
            if hasattr(unwrapped, "save_pretrained"):
                adapter_path = os.path.join(self.checkpoint_dir, f"lora_adapter_epoch_{epoch+1}")
                unwrapped.save_pretrained(adapter_path)

            try:
                from peft import get_peft_model_state_dict
                lora_sd = get_peft_model_state_dict(unwrapped)
            except Exception:
                lora_sd = {k: v for k, v in model_sd.items() if "lora" in k.lower()}

            if hasattr(unwrapped, "classifier"):
                classifier_sd = unwrapped.classifier.state_dict()
            elif hasattr(unwrapped, "head"):
                classifier_sd = unwrapped.head.state_dict()

        elif training_strategy == "full":
            full_model_sd = model_sd  # capture entire model

        state = {
            "epoch": epoch + 1,
            "best_val_acc": getattr(self, "best_val_acc", None),
            "epochs_without_improvement": getattr(self, "epochs_without_improvement", None),
            "training_strategy": training_strategy,
            "config": {
                "lr": self.lr,
                "batch_size": self.batch_size,
                "num_classes": self.global_config.get("NUM_CLASSES", 0),
                "freeze_backbone": training_strategy,
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state().cpu(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        if training_strategy == "head_only":
            state["classifier_state_dict"] = classifier_sd
        elif training_strategy == "lora":
            state["lora_state_dict"] = lora_sd
            state["classifier_state_dict"] = classifier_sd
            state["adapter_path"] = adapter_path
        elif training_strategy == "full":
            state["model_state_dict"] = full_model_sd

        return state

    def save_checkpoint(self, optimizer, epoch, is_best=False, scheduler=None, scaler=None):
        if not self.accelerator.is_main_process:
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
     

        ckpt = self._create_checkpoint_data(optimizer, epoch, scheduler, scaler)
        
        ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        self.accelerator.save(ckpt, ckpt_path)

        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
            hf_dir = os.path.join(self.checkpoint_dir, f"hf_model_epoch_{epoch+1}")
            self.accelerator.save_model(unwrapped, hf_dir)
        except Exception as e:
            print(f"[WARN] Skipped HF model save: {e}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            self.accelerator.save(ckpt, best_path)

            strategy = self.global_config.get("TRAINING_STRATEGY")
            if strategy == "lora":
                try:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    best_adapter = os.path.join(self.checkpoint_dir, "best_lora_adapter")
                    unwrapped.save_pretrained(best_adapter)
                except Exception as e:
                    print(f"Warning: Could not save best LoRA adapter: {e}")

            elif strategy == "full":
                try:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    best_full = os.path.join(self.checkpoint_dir, "best_full_model")
                    self.accelerator.save_model(unwrapped, best_full)
                except Exception as e:
                    print(f"Warning: Could not save best full-model dir: {e}")

        print(f"Checkpoint saved: {ckpt_path}")