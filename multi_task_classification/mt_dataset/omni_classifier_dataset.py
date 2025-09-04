import numpy as np
import os
import sys
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from verl.utils.dataset.rl_dataset import RLHFDataset

class OmniClassifierDataset(RLHFDataset):
    def __init__(self, *args, label_key='answer', label_map=None, dataset_key='dataset', **kwargs):
        super().__init__(*args, **kwargs)
        self.label_key = label_key 
        self.label_map = label_map  # Optional: dict mapping raw label to class index
        self.dataset_key = dataset_key

    def __getitem__(self, item):
        # connects to RLHFDataset __getitem__
        # gets the row dict which includes processed inputs 
        # from Qwen etc.
        row_dict = super().__getitem__(item)
        
        # -------------------------------------------------------------------
        # NOTE: ONLY FOR DEBUGGING, remove after done.
        # row_dict: dict = self.dataframe[item]
        # if "input_ids" not in row_dict:
        #     row_dict["input_ids"] = torch.randint(
        #         low=0, high=1000, size=(32,), dtype=torch.long)  # e.g., seq_len=32
        # if "attention_mask" not in row_dict:
        #     row_dict["attention_mask"] = torch.ones_like(row_dict["input_ids"])


        # LOADING OF VIDEO/AUDIO FEATURES
        # -------------------------------------------------------------------
        video_feats_path = row_dict.get('ext_video_feats_path', None)
        audio_feats_path = row_dict.get('ext_audio_feats_path', None)

    
        video_feat = torch.load(video_feats_path) if video_feats_path else None
    
        audio_feat = torch.load(audio_feats_path) if audio_feats_path else None
        row_dict['video_feats'] = video_feat
        row_dict['audio_feats'] = audio_feat
        # -------------------------------------------------------------------

        # --- your existing label extraction ---
        raw_label = row_dict.get(self.label_key, "").lower()
        dataset_name = row_dict.get(self.dataset_key, "").lower()  # Get dataset name from row
        
        # Construct the full label key: dataset_name_raw_label
        if dataset_name:
            full_label_key = f"{dataset_name}_{raw_label}"
        else:
            # Fallback to just raw_label if no dataset name is available
            full_label_key = raw_label

        # make sure that the label_map is all in lowercase
        if self.label_map is not None:
            self.label_map = {k.lower(): v for k, v in self.label_map.items()}
            label = self.label_map.get(full_label_key, 0)
            
            # Debug print to help with troubleshooting
            if label == 0 and full_label_key not in self.label_map:
                print(f"[WARN] Label key '{full_label_key}' not found in label map.")
                print(f"[WARN] Raw label: '{raw_label}', Dataset: '{dataset_name}'")
                print(f"[WARN] Available keys (first 10): {list(self.label_map.keys())[:10]}...")
                raise ValueError(f"Label key '{full_label_key}' not found in label map.")
        else:
            raise ValueError(f"label_map must be provided for mapping raw labels {full_label_key} to class indices")

        # --- CRITICAL: ensure tensors so collate stacks them as tensors ---
        # labels must be Long for CrossEntropyLoss
        row_dict["labels"] = torch.tensor(label, dtype=torch.long)

        return row_dict
