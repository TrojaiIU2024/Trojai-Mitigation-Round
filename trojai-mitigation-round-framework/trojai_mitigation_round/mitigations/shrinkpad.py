# Inspired from Backdoorbox
# https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/ShrinkPad.py

from typing import Dict
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from imagecorruptions import corrupt

from trojai_mitigation_round.mitigations import TrojAIMitigation, TrojAIMitigatedModel


class GaussianBlurMitigation(TrojAIMitigation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess_transform(self, x):
        original_batch_size = x.shape[0]
        x_corrupted = []
        for image in x:
            image_np = image.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype('uint8') 
            # Apply Gaussian blur corruption
            corrupted_image_np = corrupt(image_np, corruption_name='zoom_blur', severity=1)
            corrupted_image = torch.tensor(corrupted_image_np).float() / 255.0
            corrupted_image = corrupted_image.permute(2, 0, 1)  
            x_corrupted.append(corrupted_image)
        
        x_corrupted = torch.stack(x_corrupted)
        return x_corrupted, {"original_batch_size": original_batch_size}

    def mitigate_model(self, model: torch.nn.Module, dataset: Dataset) -> TrojAIMitigatedModel:
        return TrojAIMitigatedModel(model.state_dict())


