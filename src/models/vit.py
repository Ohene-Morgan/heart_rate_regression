import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification

def create_hf_vit_regression():
    # 1. Load the configuration for the base model
    model_name = 'google/vit-base-patch16-224'
    config = ViTConfig.from_pretrained(model_name)

    # 2. Tell the config we only want 1 output (Heart Rate) and 4 channels
    config.num_labels = 1
    config.num_channels = 4

    # 3. Load the model with this config
    model = ViTForImageClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )

    # 4. Modify the first layer (Patch Embeddings) for 4 channels
    old_embeddings = model.vit.embeddings.patch_embeddings.projection
    model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
        in_channels=4,
        out_channels=old_embeddings.out_channels,
        kernel_size=old_embeddings.kernel_size,
        stride=old_embeddings.stride
    )

    # 5. Initialize weights for the 4th channel (Copying Red channel weights)
    with torch.no_grad():
        new_weight = model.vit.embeddings.patch_embeddings.projection.weight.clone()
        new_weight[:, 3, :, :] = old_embeddings.weight[:, 0, :, :]
        model.vit.embeddings.patch_embeddings.projection.weight = nn.Parameter(new_weight)

    return model