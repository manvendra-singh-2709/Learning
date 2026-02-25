import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import radius_graph
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# DATASET
# ============================
class AtomDataset(Dataset):
    def __init__(self, structures):
        super().__init__()
        self.structures = structures

        # --- compute energy normalization ---
        energies = [s.info["REF_energy"] / len(s) for s in structures]
        self.e_mean = float(np.mean(energies))
        self.e_std = float(np.std(energies) + 1e-8)
        print(f"Energy mean (eV/atom): {self.e_mean:.4f}")
        print(f"Energy std  (eV/atom): {self.e_std:.4f}")

    def len(self):
        return len(self.structures)

    def get(self, idx):
        s = self.structures[idx]

        pos = torch.tensor(s.positions, dtype=torch.float32)
        z = torch.tensor(s.numbers, dtype=torch.long)

        energy = torch.tensor(
            [(s.info["REF_energy"] / len(s) - self.e_mean) / self.e_std],
            dtype=torch.float32,
        )

        forces = torch.tensor(s.arrays["REF_forces"], dtype=torch.float32)

        data = Data(
            pos=pos,
            z=z,
            y=energy,
            forces=forces,
        )
        return data


# ============================
# MODEL
# ============================
class EquivariantBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return x + self.mlp(x)  # residual


class EquivariantGNN(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=8, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff

        self.embedding = nn.Embedding(100, hidden_dim)

        self.layers = nn.ModuleList(
            [EquivariantBlock(hidden_dim) for _ in range(num_layers)]
        )

        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        pos = data.pos.requires_grad_(True)
        z = data.z

        edge_index = radius_graph(pos, r=self.cutoff, loop=False)

        x = self.embedding(z)

        # deep residual stack
        for layer in self.layers:
            x = layer(x)

        # global pooling
        x_mean = x.mean(dim=0)

        energy = self.energy_head(x_mean).squeeze()

        # Forces = -∂E/∂R
        forces = -torch.autograd.grad(
            energy,
            pos,
            create_graph=True,
            retain_graph=True,
        )[0]

        return energy, forces


# ============================
# TRAIN
# ============================
def train(model, loader, optimizer):
    model.train()

    loss_e = nn.MSELoss()
    loss_f = nn.MSELoss()
    loss_s = nn.MSELoss()

    total_loss = 0

    for batch in tqdm(loader, desc="Epoch"):
        batch = batch.to(device)

        optimizer.zero_grad()

        pred_e, pred_f, pred_s = model(batch)

        # ensure shapes match
        pred_e = pred_e.view(1)

        le = loss_e(pred_e, batch.y)
        lf = loss_f(pred_f, batch.forces)

        # balanced loss (VERY IMPORTANT)
        loss = le + 0.1 * lf + 0.01

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================
# PARAM COUNT
# ============================
def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================
# MAIN
# ============================
def main():
    # ---- load your ASE structures here ----
    from ase.io import read
    structures = read("data/train.extxyz", ":")

    dataset = AtomDataset(structures)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = EquivariantGNN(hidden_dim=256, num_layers=8).to(device)

    print(f"\nTotal parameters: {count_params(model):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)

    for epoch in range(1, 31):
        loss = train(model, loader, optimizer)
        print(f"Epoch {epoch} Loss: {loss:.6f}")


if __name__ == "__main__":
    main()