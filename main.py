import os
import torch
from car_dataset import CarMeshDataset
from torch.utils.data import DataLoader
from trainer import Trainer
from model import DiffusionNet
from cache import prepopulate_cache
from utils import create_fold_splits

from config import Config

torch.manual_seed(Config.seed)

# Cache data
print("Caching data (might take a while)...")
# prepopulate_cache(
#     Config.data_basepath,
#     cache_dir=Config.cache_dir,
#     n_eig=Config.num_eig,
#     n_workers=Config.n_workers,
# )

for fold in range(Config.n_folds):
    # Create fold splits
    print("Creating splits...")
    create_fold_splits(
        os.path.join(Config.data_basepath, "cache"),
        os.path.join(Config.data_basepath, "splits.json"),
        n_folds=Config.n_folds,
        ratio=Config.train_ratio,
        seed=Config.seed,
        n_eig=Config.num_eig,
    )  # Will save a splits.json file in data_basepath

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataset = CarMeshDataset(
        Config.data_basepath,
        fold=fold,
        train=True,
        n_eig=Config.num_eig,
        device=Config.device,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
    )

    valid_dataset = CarMeshDataset(
        Config.data_basepath,
        fold=fold,
        train=False,
        n_eig=Config.num_eig,
        device=Config.device,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=None,
        num_workers=0,
        persistent_workers=False,
    )

    # Create trainer
    model_cfg = {
        "inp_feat": Config.inp_feat,
        "num_eig": Config.num_eig,
        "p_in": Config.p_in,
        "p_out": Config.p_out,
        "N_block": Config.n_block,
        "n_channels": Config.n_channels,
        "outputs_at": Config.outputs_at
    }

    my_trainer = Trainer(
        DiffusionNet,
        model_cfg,
        train_loader,
        valid_loader,
        device=Config.device,
        save_dir=Config.save_dir,
        figures_dir=Config.figures_dir,
        log_wandb=Config.log_wandb,
        num_epochs=Config.num_epochs,
    )

    # Run train
    my_trainer.run()