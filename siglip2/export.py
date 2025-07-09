import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoProcessor, SiglipVisionConfig
from torch.jit._trace import is_tracing


def torch_int(x):
    """
    Casts an input to a torch int64 tensor if we are in a tracing context, otherwise to a Python int.
    """
    return x.to(torch.int64) if is_tracing() and isinstance(x, torch.Tensor) else int(x)


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing and no class embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        if pixel_values.ndim == 4:
            if isinstance(self.patch_embedding, nn.Conv2d):
                _, _, height, width = pixel_values.shape
                target_dtype = self.patch_embedding.weight.dtype
                patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
                embeddings = patch_embeds.flatten(2).transpose(1, 2)
            elif isinstance(self.patch_embedding, nn.Linear):
                batch, channel, height, width = pixel_values.shape
                target_dtype = self.patch_embedding.weight.dtype
                pixel_values = pixel_values.reshape(batch, channel, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size)
                pixel_values = pixel_values.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channel * self.patch_size * self.patch_size)
                patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
                embeddings = patch_embeds.view(batch, -1, self.embed_dim)
            else:
                raise ValueError(f"Unsupported patch_embedding type: {type(self.patch_embedding)}")
        elif pixel_values.ndim == 3:
            _, seqlen, hidden_size = pixel_values.shape
            height = width = int(seqlen**0.5)
            target_dtype = self.patch_embedding.weight.dtype
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            embeddings = patch_embeds.view(-1, seqlen, hidden_size)
        else:
            raise ValueError(f"Unsupported pixel_values shape: {pixel_values.shape}")

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


def convert_conv_to_linear(model: nn.Module):
    """
    Converts convolutional layers in the model to linear layers.
    """
    model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Get the input and output dimensions
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]
            padding = module.padding

            assert stride == kernel_size, "Stride must be equal to kernel size for conversion to linear layer."
            if not isinstance(padding, str):
                assert padding[0] == 0, "Padding must be zero for conversion to linear layer."

            # Create a new linear layer
            linear_layer = nn.Linear(
                in_features=in_channels * kernel_size * kernel_size,
                out_features=out_channels,
            )

            # Copy the weights from the conv layer to the linear layer
            with torch.no_grad():
                linear_layer.weight.copy_(module.weight.view(out_channels, -1))
                if module.bias is not None:
                    linear_layer.bias.copy_(module.bias)

            # Replace the conv layer with the linear layer
            model._modules[name] = linear_layer

    return model


if __name__ == "__main__":
    torch.manual_seed(42)

    ckpt = "google/siglip2-base-patch16-384"
    model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
    processor = AutoProcessor.from_pretrained(ckpt)

    ori_conv_embeddings = model.vision_model.embeddings
    conv_embeddings = SiglipVisionEmbeddings(ori_conv_embeddings.config)
    conv_embeddings.load_state_dict(ori_conv_embeddings.state_dict())
    linear_embeddings = convert_conv_to_linear(conv_embeddings)

    print(conv_embeddings)
    print(linear_embeddings)

    height = width = 384
    pixel_values = torch.randn(1, 3, height, width)
    conv_embeddings_output = conv_embeddings(pixel_values, interpolate_pos_encoding=False)
    print("Conv embeddings output shape:", conv_embeddings_output.shape)

    patch_size = conv_embeddings.patch_size
    patch_pixel_values = pixel_values.reshape(1, 3, height // patch_size, patch_size, width // patch_size, patch_size)
    patch_pixel_values = patch_pixel_values.permute(0, 2, 4, 1, 3, 5).reshape(1, -1, 3 * patch_size * patch_size)
    linear_embeddings_output = linear_embeddings(patch_pixel_values, interpolate_pos_encoding=False)
    print("Linear embeddings output shape:", linear_embeddings_output.shape)

    assert conv_embeddings_output.shape == linear_embeddings_output.shape, "Output shapes do not match after conversion."

    diff = torch.abs(conv_embeddings_output - linear_embeddings_output)
    print("Max difference between conv and linear embeddings:", diff.max().item())
    print("Mean difference between conv and linear embeddings:", diff.mean().item())
    print("Conversion successful, outputs match within tolerance.")
    assert diff.max().item() < 1e-5, "Outputs do not match within tolerance after conversion."
    print("Model conversion to linear embeddings completed successfully.")

    torch.onnx.export(
        conv_embeddings,
        pixel_values,
        "conv_embeddings.onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    torch.onnx.export(
        linear_embeddings,
        patch_pixel_values,
        "linear_embeddings.onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    torch.onnx.export(
        linear_embeddings,
        pixel_values,
        "linear_embeddings_image_input.onnx",
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
