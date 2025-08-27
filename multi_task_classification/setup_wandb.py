#!/usr/bin/env python3
"""
Helper script to set up wandb for the OmniClassifier training.
"""

import os
import wandb

def setup_wandb():
    """Set up wandb configuration."""
    print("Setting up wandb for OmniClassifier training...")
    
    # Check if wandb is logged in
    try:
        api = wandb.Api()
        print("✓ Wandb is already configured and logged in.")
        return True
    except Exception as e:
        print("✗ Wandb not configured. Please follow these steps:")
        print("\n1. Install wandb if not already installed:")
        print("   pip install wandb")
        print("\n2. Login to wandb:")
        print("   wandb login")
        print("\n3. Or set your API key as an environment variable:")
        print("   export WANDB_API_KEY=your_api_key_here")
        print("\n4. You can also disable wandb by setting USE_WANDB = False in train_omni_classifier.py")
        return False

def get_wandb_entity():
    """Get the current wandb entity (username)."""
    try:
        api = wandb.Api()
        return api.default_entity
    except:
        return None

if __name__ == "__main__":
    setup_wandb()
    entity = get_wandb_entity()
    if entity:
        print(f"Current wandb entity: {entity}")
        print("You can set WANDB_ENTITY in train_omni_classifier.py to use a different entity.")
