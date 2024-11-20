import os
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as plu
from mesh_utils.mesh import TriMesh
from tqdm import tqdm 
from sklearn.metrics import r2_score
import wandb

class Trainer(object):

    def __init__(self, diffusionnet_cls, model_cfg, train_loader, valid_loader, device='cuda',
                 lr=1e-3, weight_decay=1e-4, num_epochs=200,
                 lr_decay_every = 50, lr_decay_rate = 0.5,
                 log_interval=1, save_dir=None, figures_dir=None, log_wandb=False):

        """
        diffusionnet_cls: (nn.Module) class of the DiffusionNet model
        model_cfg: (dict) keyword arguments for model
        train_loader: (torch.utils.DataLoader) DataLoader for training set
        valid_loader: (torch.utils.DataLoader) DataLoader for validation set
        device: (str) 'cuda' or 'cpu'
        lr: (float) learning rate
        weight_decay: (float) weight decay for optimiser
        num_epochs: (int) number of epochs
        lr_decay_every: (int) decay learning rate every this many epochs
        lr_decay_rate: (float) decay learning rate by this factor
        log_interval: (int) print training stats every this many iterations
        save_dir: (str) directory to save model checkpoints
        figures_dir: (str) directory to save figures
        log_wandb: (bool) log wandb or not 
        """

        self.model = diffusionnet_cls(
            C_in=model_cfg['p_in'],
            C_out=model_cfg['p_out'],
            N_block=model_cfg['N_block'],
            outputs_at=model_cfg['outputs_at'],
        )


        self.loss = nn.MSELoss()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.lr_decay_every = lr_decay_every
        self.lr_decay_rate = lr_decay_rate
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.figures_dir = figures_dir

        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []

        self.inp_feat = model_cfg.get('inp_feat', 'xyz')
        self.num_eig = model_cfg.get('num_eig', 128)
        if not self.inp_feat in ['xyz', 'hks', 'wks']:
            raise ValueError('inp_feat must be one of xyz, hks, wks')

        self.model.to(self.device)

        self.log_wandb = log_wandb
        if self.log_wandb:
            wandb.init(project='DragDiffNet', config=model_cfg)
            wandb.watch(self.model)

    def forward_step(self, verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY):
        """
        Perform a forward step of the model.

        Args:
            verts (torch.Tensor): (N, 3) tensor of vertex positions
            faces (torch.Tensor): (F, 3) tensor of face indices
            frames (torch.Tensor): (N, 3, 3) tensor of tangent frames.
            vertex_area (torch.Tensor): (N, N) sparse Tensor of vertex areas.
            L (torch.Tensor): (N, N) sparse Tensor of cotangent Laplacian.
            evals (torch.Tensor): (num_eig,) tensor of eigenvalues.
            evecs (torch.Tensor): (N, num_eig) tensor of eigenvectors.
            gradX (torch.Tensor): (N, N) tensor of gradient in X direction.
            gradY (torch.Tensor): (N, N) tensor of gradient in Y direction.

        Returns:
            pred (torch.Tensor): (N, p_out) tensor of predicted labels.
        """

        if self.inp_feat == 'xyz':
            features = verts
        elif self.inp_feat == 'hks':
            features = self.compute_HKS(verts, faces, self.num_eig, n_feat=32)
        elif self.inp_feat == 'wks':
            features = self.compute_WKS(verts, faces, self.num_eig, num_E=32)

        preds = self.model(features, vertex_area, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

        return preds



    def train_epoch(self):
        """
        Train the network for one epoch
        """
        train_loss = 0
        all_preds = []
        all_labels = []
        for i, batch in enumerate(tqdm(self.train_loader, "Train epoch")):

            verts = batch["vertices"].to(self.device)
            faces = batch["faces"].to(self.device)
            frames = batch["frames"].to(self.device)
            vertex_area = batch["vertex_area"].to(self.device)
            L = batch["L"].to(self.device)
            evals = batch["evals"].to(self.device)
            evecs = batch["evecs"].to(self.device)
            gradX = batch["gradX"].to(self.device)
            gradY = batch["gradY"].to(self.device)
            label = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            preds = self.forward_step(verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY)
            loss = self.loss(preds, label).float()

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # Collect predictions and labels for R2 score
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(label.detach().cpu().numpy()[None])

        # Compute R2 score
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        train_r2 = r2_score(all_labels, all_preds)

        return train_loss/len(self.train_loader), train_r2

    def valid_epoch(self):
        """
        Run a validation epoch
        """
        val_loss = 0
        all_preds = []
        all_labels = []
        print("Start val epoch")
        for i, batch in enumerate(self.valid_loader):

            # READ BATCH
            verts = batch["vertices"].to(self.device)
            faces = batch["faces"].to(self.device)
            frames = batch["frames"].to(self.device)
            vertex_area = batch["vertex_area"].to(self.device)
            L = batch["L"].to(self.device)
            evals = batch["evals"].to(self.device)
            evecs = batch["evecs"].to(self.device)
            gradX = batch["gradX"].to(self.device)
            gradY = batch["gradY"].to(self.device)
            labels = batch["label"].to(self.device)

            preds = self.forward_step(verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY)
            loss = self.loss(preds, labels)

            val_loss += loss.item()

            # Collect predictions and labels for R2 score
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy()[None])

        # Compute R2 score
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        val_r2 = r2_score(all_labels, all_preds)

        print("End val epoch")
        return val_loss/len(self.valid_loader), val_r2, all_labels, all_preds

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)
        for epoch in range(self.num_epochs):
            self.model.train()

            if epoch % self.lr_decay_every == 0:
                self.adjust_lr()

            train_ep_loss, train_r2 = self.train_epoch()
            self.train_losses.append(train_ep_loss)

            if epoch % self.log_interval == 0:
                val_loss, val_r2, all_labels, all_preds = self.valid_epoch()
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_latest.pth'))
                print(f'Epoch: {epoch:03d}/{self.num_epochs}, '
                      f'Train Loss: {train_ep_loss:.4f}, '
                      f'Train R2: {train_r2:.4f}, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Val R2: {val_r2:.4f}')
                # Log metrics to wandb
                if self.log_wandb:
                    wandb.log({
                        'train_loss': train_ep_loss,
                        'train_r2': train_r2,
                        'val_loss': val_loss,
                        'val_r2': val_r2,
                    })

        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_final.pth'))

        self.plot_pred_vs_sim(self, all_labels, all_preds)

    def plot_pred_vs_sim(self, all_labels, all_preds):
        os.makedirs(self.figures_dir, exist_ok=True)

        plt.figure(figsize=(8, 8))
        plt.scatter(all_labels, all_preds, alpha=0.6, color='blue', edgecolor='k', s=10)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # Diagonal reference line
        plt.xlabel('Simulated drag coefficient')
        plt.ylabel('Predicted drag coefficient')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Predicted vs Simulated Drag Coefficient')

        # Save plot in figures directory
        figure_path = os.path.join(self.figures_dir, 'plot_pred_vs_sim.png')
        plt.savefig(figure_path)
        plt.close()


    def visualize(self, i=0, j=1):
        """
        We only test two shapes of validation set.
        """
        self.model.eval()
        test_seg_meshes = []
        labels = []
        verts = []
        for i, batch in enumerate(self.valid_loader):
            verts.append(batch["vertices"])
            labels.append(batch["label"])
            test_seg_meshes.append(TriMesh(batch["vertices"], batch["faces"]))

            if i==10:
                break

        plu.double_plot(test_seg_meshes[i], test_seg_meshes[j])
        print(f"Cx value: {labels[i]}, {labels[j]}")
        print(f"Number of verts: {verts[i].shape[0]}, {verts[j].shape[0]}")

    def adjust_lr(self):
        lr = self.lr * self.lr_decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_HKS(self, evecs, evals, num_eig, n_feat):
        """
        Compute the HKS features for each vertex in the mesh.
        Args:
            evecs (torch.Tensor): (N, K) tensor of eigenvectors
            evals (torch.Tensor): (K,) tensor of eigenvectors
            num_eig (int): number of eigenvalues to use
            n_feat (int): number of features to compute

        Returns:
            hks (torch.Tensor): (N, n_feat) tensor of HKS features
        """
        abs_ev = torch.sort(torch.abs(evals)).values[:num_eig].detach().cpu()

        t_list = np.geomspace(4*np.log(10)/abs_ev[-1], 4*np.log(10)/abs_ev[1], n_feat)
        t_list = torch.Tensor(t_list.astype(np.float32))

        evals_s = abs_ev

        coefs = torch.exp(-t_list[:,None] * evals_s[None,:])  # (num_T,K)

        natural_HKS = np.einsum('tk,nk->nt', coefs, evecs.detach().cpu()[:,:num_eig].square())

        inv_scaling = coefs.sum(1)  # (num_T)

        return ((1/inv_scaling)[None,:] * natural_HKS).to(device=evecs.device)

    def compute_WKS(self, evecs, evals, num_eig, n_feat):
        """
        Compute the WKS features for each vertex in the mesh.

        Args:
            evecs (torch.Tensor): (N, K) tensor of eigenvectors
            evals (torch.Tensor): (K,) tensor of eigenvectors
            num_eig (int): number of eigenvalues to use
            n_feat (int): number of features to compute

        Returns:
            wks: torch.Tensor: (N, num_E) tensor of WKS features
        """
        abs_ev = torch.sort(torch.abs(evals)).values[:num_eig]

        e_min,e_max = np.log(abs_ev[1]),np.log(abs_ev[-1])
        sigma = 7*(e_max-e_min)/n_feat

        e_min += 2*sigma
        e_max -= 2*sigma

        energy_list = torch.linspace(e_min,e_max,n_feat)

        evals_s = abs_ev

        coefs = torch.exp(-torch.square(energy_list[:,None] - torch.log(torch.abs(evals_s))[None,:])/(2*sigma**2))  # (num_E,K)

        natural_WKS = np.einsum('tk,nk->nt', coefs, evecs[:,:num_eig].square())

        inv_scaling = coefs.sum(1)  # (num_E)
        return (1/inv_scaling)[None,:] * natural_WKS

