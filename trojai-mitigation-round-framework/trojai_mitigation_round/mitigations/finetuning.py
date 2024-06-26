from typing import Dict
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from trojai_mitigation_round.mitigations.mitigation import TrojAIMitigation
from trojai_mitigation_round.mitigations.mitigated_model import TrojAIMitigatedModel

class RandomizedDataset(Dataset):
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, label, fname = self.dataset[idx]
        randomized_label = random.randint(0, self.num_classes - 1)
        return x, randomized_label, fname

class SEAMTrojai(TrojAIMitigation):
    def __init__(self, device, loss_cls, optim_cls, lr, epochs, ckpt_dir="./ckpts", ckpt_every=0, batch_size=32, num_workers=1, **kwargs):
        super().__init__(device, batch_size, num_workers, **kwargs)
        self._optimizer_class = optim_cls
        self._loss_cls = loss_cls
        self.lr = lr
        self.epochs = epochs
        self.ckpt_dir = ckpt_dir
        self.ckpt_every = ckpt_every

    def train_model(self, model, dataset, description):
        """
        Trains the model on the given dataset.
        """
        model = model.to(self.device)
        model.train()
        optim = self._optimizer_class(model.parameters(), lr=self.lr)
        loss_fn = self._loss_cls()
        trainloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        
        for i in range(self.epochs):
            pbar = tqdm(trainloader)
        
            for x, y, _ in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                optim.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optim.step()
                pbar.set_description(f"Epoch: {i} | Loss: {loss}")
            
            if self.ckpt_every != 0 and i % self.ckpt_every == 0:
                ckpt_path = Path(self.ckpt_dir)
                ckpt_path.mkdir(exist_ok=True)
                torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                }, ckpt_path / Path(f"ft_ckpt_epoch{i + 1}.ckpt"))
                print(f"Saved ckpt to {ckpt_path / Path(f'ft_ckpt_epoch{i + 1}.ckpt')}")

        return model

    def mitigate_model(self, model: torch.nn.Module, dataset: Dataset) -> TrojAIMitigatedModel:
        """
        Args:
            model: the model to repair
            dataset: a dataset of examples
        Returns:
            mitigated_model: A TrojAIMitigatedModel object corresponding to new model weights and a pre/post processing techniques
        """
        # Determine the number of classes from the model's output dimension
        num_classes = model(torch.rand(1, *next(iter(dataset))[0].shape).to(self.device)).shape[1]
        
        # Step 1: Induce Catastrophic Forgetting
        randomized_dataset = RandomizedDataset(dataset, num_classes)
        model = self.train_model(model, randomized_dataset, "Inducing Forgetting")
        
        # Step 2: Recover the Primary Task
        model = self.train_model(model, dataset, "Recovering Primary Task")
        
        return TrojAIMitigatedModel(model)
