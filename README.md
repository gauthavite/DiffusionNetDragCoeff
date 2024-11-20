# DragDiffNet

This repository implements a custom version of **DiffusionNet**, inspired by the paper ["DiffusionNet: Discretization Agnostic Learning on Surfaces"](https://arxiv.org/abs/2206.09398).

## Dataset

We use the [Decode Project dataset](https://decode.mit.edu/projects/dragprediction/), which contains:
- 2,474 high-quality car meshes from ShapeNet
- Drag coefficients calculated using Computational Fluid Dynamics (CFD).

## Objective

The goal is to develop a model that accurately predicts drag coefficients directly from 3D meshes. Unlike the original paper, which projects 3D meshes onto 2D images for this task, our approach aims to achieve strong predictive performance directly on the 3D geometry.

## Methodology

We leverage **DiffusionNet**, a sampling- and resolution-agnostic model that operates directly on 3D meshes. It uses a learned diffusion layer to extract meaningful geometric features, enabling effective drag coefficient prediction.

## How to Run

1. Update the configuration parameters in `config.py` as needed.
2. Run the following command to start training or testing:
   ```bash
   python main.py
