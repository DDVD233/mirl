# label_maps.py
"""
Central place for label mappings for OmniClassifierDataset.
Each dict maps raw string labels → integer indices.
"""

# 5-class sentiment labels
SENTIMENT_LABEL_MAP = {
    "negative": 0,
    "weakly negative": 1,
    "neutral": 2,
    "weakly positive": 3,
    "positive": 4,
}

# 6-class emotion labels
EMOTION_LABEL_MAP = {
    "happy": 0,
    "sad": 1,
    "anger": 2,
    "surprise": 3,
    "disgust": 4,
    "fear": 5,
}

## All LABEL MAPS