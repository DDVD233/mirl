# MIRL: Multisensory Intelligence Reinforcement Learning for LLMs

MIRL is a flexible reinforcement learning framework for training large language models with multimodal capabilities. This framework is built upon [verl](https://github.com/volcengine/verl), extending its capabilities to support diverse modalities and annotation formats.

## Key Features

### 🎯 Enhanced Annotation Support
- Support for multiple annotation formats beyond standard text
- Native support for Geometry3k format for mathematical reasoning tasks
- Flexible annotation pipeline for custom formats

### 🌐 Multimodal Training
- **Audio support** (implemented): Train models with audio understanding and generation capabilities
- **Extensible architecture**: Framework designed to accommodate arbitrary modalities
- Active development for additional modality support

### 🚀 Future Roadmap
- **Diffusion Language Models**: Planned support for training diffusion-based language models
- Unified training pipeline for both autoregressive and diffusion architectures

## Getting Started

### Prerequisites
- CUDA-compatible GPU (recommended: A100, H100, or similar)
- CUDA 12.1 or higher
- Python 3.10 - 3.12

### Installation

1. **Create a new conda environment**
   ```bash
   conda create -n mirl python=3.11
   conda activate mirl
   ```

2. **Install uv and vLLM**
   ```bash
   pip install uv
   uv pip install vllm --torch-backend=auto
   ```

3. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Flash Attention**
   ```bash
   git clone https://github.com/Dao-AILab/flash-attention
   cd flash-attention
   MAX_JOBS=16 python setup.py install
   ```

## Acknowledgments

MIRL is a fork of [verl (Volcano Engine Reinforcement Learning)](https://github.com/volcengine/verl), which provides the foundational HybridFlow framework and efficient RLHF training infrastructure.

## License

This project inherits the Apache 2.0 License from the original verl framework.