import torch
import torch.nn
import torchvision as tv
from torchvision.models.feature_extraction import create_feature_extractor

class EffnetModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        effnet = tv.models.efficientnet_v2_s()
        self.model = create_feature_extractor(effnet, ["flatten"])
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1280,7),
        )
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1280,7),
        )
    
    def forward(self, x):
        x = self.model(x)["flatten"]
        return self.nn_fracture(x), self.nn_vertebrae(x)

    def predict(self, x):
        frac, vert = self.forward(x)
        return torch.sigmoid(frac), torch.sigmoid(vert)



