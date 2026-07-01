import torch.nn as nn
import torchvision.models as models


def build_model():
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )
    return model


def unfreeze_top_layers(model, n_blocks=3):
    for block_idx in range(9 - n_blocks, 9):
        for param in model.features[block_idx].parameters():
            param.requires_grad = True
