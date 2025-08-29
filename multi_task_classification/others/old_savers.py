
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


    def _find_latest_checkpoint(self):
        """Find any .pt file in the checkpoint directory."""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoint_files = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith(".pt"):
                checkpoint_files.append(os.path.join(self.checkpoint_dir, file))
        
        if not checkpoint_files:
            return None
        
        # Return the first .pt file found (or you could implement more sophisticated logic)
        return checkpoint_files[0]

    def _load_model_state_from_checkpoint(self, checkpoint, context="checkpoint"):
        """
        Load model weights according to the training strategy saved in `checkpoint`.
        - head_only:   loads only classifier/head module params
        - lora:        loads PEFT adapters (preferred) + classifier/head params
        - full:        loads full model state dict
        Fallbacks handle missing artifacts gracefully.
        """
        
        # ASSUME THAT THE MODEL IS ALREADY UNWRAPPED
        training_strategy = checkpoint.get("training_strategy", "head_only")
        # with open("/home/keaneong/human-behavior/verl/multi_task_classification/classifier_state_dict_KEY.txt", "a") as f:
        #     f.write(f"Classifier state dict {checkpoint['classifier_state_dict']}")
        # raise Exception("Stop here")
        
        def _report_load(res):
            if isinstance(res, tuple):
                missing, unexpected = res
            else:
                missing, unexpected = res.missing_keys, res.unexpected_keys
            if missing:
                print(f"[load:{context}] missing keys: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
            if unexpected:
                print(f"[load:{context}] unexpected keys: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

        if training_strategy == "head_only" and "classifier_state_dict" in checkpoint:
            print(f"Loading head-only {context} (classifier/head only)...")

            res = self.model.classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=False)
            _report_load(res)

        elif training_strategy == "lora":
            print(f"Loading LoRA {context} (adapters + optional classifier/head)...")

            # 1) Load classifier/head if present
            cls_sd = checkpoint.get("classifier_state_dict")
            if cls_sd is not None:
                if hasattr(self.model, "classifier"):
                    _report_load(self.model.classifier.load_state_dict(cls_sd, strict=False))
                elif hasattr(self.model, "head"):
                    _report_load(self.model.head.load_state_dict(cls_sd, strict=False))
                else:
                    _report_load(self.model.load_state_dict(cls_sd, strict=False))

            # 2) Prefer loading adapters from a saved PEFT directory
            adapter_path = checkpoint.get("adapter_path")
            if not adapter_path:
                # Try to find one in the checkpoint dir: best first, then epoch dirs
                ckpt_dir = os.path.dirname(getattr(self, "load_checkpoint_path", "") or self.checkpoint_dir)
                best_dir = os.path.join(ckpt_dir, "best_lora_adapter")
                if os.path.isdir(best_dir):
                    adapter_path = best_dir
                else:
                    # find most recent 'lora_adapter_epoch_*'
                    candidates = [os.path.join(ckpt_dir, d)
                                for d in os.listdir(ckpt_dir) if "lora_adapter" in d]
                    if candidates:
                        adapter_path = sorted(candidates)[-1]

            loaded_adapter = False
            try:
                # If the model exposes PEFT's load_adapter, use it.
                if adapter_path and os.path.isdir(adapter_path) and hasattr(self.model, "load_adapter"):
                    print(f"Loading LoRA adapter directory: {adapter_path}")
                    self.model.load_adapter(adapter_path)
                    loaded_adapter = True
            except Exception as e:
                print(f"[WARN] PEFT adapter load failed from dir: {e}")

            # 3) Fallback: direct param load (when no adapter directory is available)
            if not loaded_adapter:
                lora_sd = checkpoint.get("lora_state_dict")
                if lora_sd:
                    print("No adapter directory found; loading LoRA parameters directly...")
                    _report_load(self.model.load_state_dict(lora_sd, strict=False))
                else:
                    print("[WARN] No LoRA adapter dir or lora_state_dict found—skipping adapter load.")

        elif training_strategy == "full" and "model_state_dict" in checkpoint:
            print(f"Loading full {context} (entire model state)...")
            _report_load(self.model.load_state_dict(checkpoint["model_state_dict"], strict=False))

        else:
            # Last-resort compatibility path
            print(f"Loading {context} as a raw/partial state dict (strict=False)...")
            _report_load(self.model.load_state_dict(checkpoint, strict=False))

    def load_checkpoint(self, optimizer=None, scheduler=None, scaler=None):
        """
        Load the latest (or best) checkpoint for *training resume*.
        Restores model weights, optimizer, scheduler, scaler, and RNG states.
        Returns: start_epoch (int)
        """
        # Decide which .pt to read
        # only if you specify .pt for the load_checkpoint_path
        if self.load_checkpoint_path and os.path.isfile(self.load_checkpoint_path):
            target_path = self.load_checkpoint_path
        else:
            # prefer best first if asked
            target_path = self._find_latest_checkpoint()

        if not target_path:
            print("[load] No checkpoint file found; starting fresh.")
            return 0

        print(f"[load] Reading checkpoint: {target_path}")
        try:
            checkpoint = torch.load(target_path, map_location="cpu", weights_only=False)

            # 1) Model weights - load before prepare() to avoid accelerate wrapper issues
            self._load_model_state_from_checkpoint(checkpoint, context=os.path.basename(target_path))

            # 2) Optimizer/Scheduler/Scaler (if provided)
            if optimizer is not None and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if scaler is not None and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])

            # 3) Restore trainer metadata
            self.best_val_acc = checkpoint.get("best_val_acc", getattr(self, "best_val_acc", None))
            self.epochs_without_improvement = checkpoint.get(
                "epochs_without_improvement",
                getattr(self, "epochs_without_improvement", None)
            )

            # 4) RNG states (optional but good for determinism)
            rng = checkpoint.get("rng_state", None)
            if rng:
                try:
                    random.setstate(rng["python"])
                    np.random.set_state(rng["numpy"])
                    torch.set_rng_state(rng["torch"])
                    if torch.cuda.is_available() and rng["cuda"] is not None:
                        torch.cuda.set_rng_state_all(rng["cuda"])
                except Exception as e:
                    print(f"[WARN] Could not fully restore RNG state: {e}")

            start_epoch = int(checkpoint.get("epoch", 0))
            print(f"[load] Resume from epoch={start_epoch}, best_val_acc={self.best_val_acc}")
            return start_epoch

        except Exception as e:
            print(f"[WARN] Failed to load checkpoint ({target_path}): {e}. Starting fresh.")
            return 0


    def _create_checkpoint_data(self, optimizer, epoch, scheduler=None, scaler=None):
        
        training_strategy = self.global_config.get('TRAINING_STRATEGY', 'head_only')
        
        # NOTE: This only works because we have waited for everyone to finish training
        # And we have also gotten off the main process
        unwrapped = self.accelerator.unwrap_model(self.model)

        classifier_sd = None
        lora_sd = None
        adapter_path = None
        # model_sd = self.accelerator.get_state_dict(self.model)   # wrapper/sharding-safe

        if training_strategy == "head_only":
            if hasattr(unwrapped, "classifier"):
                classifier_sd = unwrapped.classifier.state_dict()
            elif hasattr(unwrapped, "head"):
                classifier_sd = unwrapped.head.state_dict()
            # else:
            #     classifier_sd = {k: v for k, v in model_sd.items()
            #                     if k.startswith("classifier.") or k.startswith("head.")}
            # with open("/home/keaneong/human-behavior/verl/multi_task_classification/classifier_states.txt", "a") as f:
            #     f.write(f"\nNew Latest Classifier state dict {classifier_sd}")
            # raise Exception("Stop here")

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
            # NOTE: PLEASE MAKE SURE THAT THIS IS A PROPERLY PRINTED STATE DICT
            model_sd = self.accelerator.get_state_dict(self.model)  # safe across wrappers
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
        # make sure the accelerator waits first
        self.accelerator.wait_for_everyone()

        # full_sd = self.accelerator.get_state_dict(self.model)   # wrapper/sharding-safe
        # with open("/home/keaneong/human-behavior/verl/multi_task_classification/debug_save_2.txt", "a") as f:
        #     f.write(f"\n=== MODEL STATE DICT DEBUG ===\n")
        #     f.write(f"Total number of parameters: {len(full_sd)}\n")
        #     f.write(f"Keys in state dict:\n")
            
        #     # Save all keys
        #     for i, key in enumerate(full_sd.keys()):
        #         f.write(f"  {i}: {key}\n")
            
        #     f.write(f"\nParameter details:\n")
        #     # Save key-value pairs with tensor info
        #     for key, value in full_sd.items():
        #         if hasattr(value, 'shape'):
        #             f.write(f"  {key}: shape={value.shape}, dtype={value.dtype}, requires_grad={value.requires_grad}\n")
        #             # Save first few values for debugging
        #             if value.numel() > 0:
        #                 flat_values = value.flatten()
        #                 f.write(f"    First 5 values: {flat_values[:5].tolist()}\n")
        #                 f.write(f"    Mean: {value.mean().item():.6f}, Std: {value.std().item():.6f}\n")
        #         else:
        #             f.write(f"  {key}: {type(value)} = {value}\n")
            
        #     f.write(f"\n=== END DEBUG ===\n")

        # raise Exception("Stop here")
        
        # if not self.accelerator.is_main_process:
        #     return

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.accelerator.save_state(self.checkpoint_dir)

        with open("/home/keaneong/human-behavior/verl/multi_task_classification/classifier_states.txt", "a") as f:
                f.write(f"\nAccelerator saved state")
                raise Exception("Stop here")


        ckpt = self._create_checkpoint_data(optimizer, epoch, scheduler, scaler)
        
        ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        self.accelerator.save(ckpt, ckpt_path)

        # with open("/home/keaneong/human-behavior/verl/multi_task_classification/debug_save.txt", "a") as f:
        #         f.write(f"\nSAVED THE CHECKPOINT")
        #         raise Exception("Stop here")

       
        # if is_best: # NOTE: Forget about this for now as it will be time consuming
        #     best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        #     self.accelerator.save(ckpt, best_path)

        #     strategy = self.global_config.get("TRAINING_STRATEGY")
        #     if strategy == "lora":
        #         try:
        #             unwrapped = self.accelerator.unwrap_model(self.model)
        #             best_adapter = os.path.join(self.checkpoint_dir, "best_lora_adapter")
        #             unwrapped.save_pretrained(best_adapter)
        #         except Exception as e:
        #             print(f"Warning: Could not save best LoRA adapter: {e}")

        #     elif strategy == "full":
        #         try:
        #             unwrapped = self.accelerator.unwrap_model(self.model)
        #             best_full = os.path.join(self.checkpoint_dir, "best_full_model")
        #             self.accelerator.save_model(unwrapped, best_full)
        #         except Exception as e:
        #             print(f"Warning: Could not save best full-model dir: {e}")

        print(f"Checkpoint saved: {ckpt_path}")