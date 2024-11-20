# DragDiffNet

This repository proposes a custom implementation of the the DiffusionNet model based on the paper "DiffusionNet: Discretization Agnostic Learning on Surfaces," accessible at [arXiv:2206.09398](https://arxiv.org/abs/2012.00888).

We'll use the dataset from https://decode.mit.edu/projects/dragprediction/. It consists of 2,474 high-quality car meshes, with their corresponding drag coefficient computed by Computational Fluid Dynamics (CFD). The goal is to make a model which accurately predicts the car drag coefficient. In their paper, it's done by projecting the 3D mesh on 2D images, but we want to try to obtain a good correlation directly from the 3D mesh.

For this, we use DiffusionNet, which is a sampling and resolution agnostic model working directly on 3D meshes by leveraging a diffusion layer with a learned.