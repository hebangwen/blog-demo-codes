import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, do_cast=True, affine=True):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.do_cast = do_cast
        self.affine = affine

    def forward(self, x):
        original_dtype = x.dtype
        # 强制转换为 float32
        if self.do_cast:
            x = x.to(torch.float32)
        # 计算均值和标准差
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + self.eps)
        # 归一化
        x_normalized = (x - mean) * rstd
        if self.affine:
            # 应用权重和偏置
            x_normalized = x_normalized * self.weight + self.bias
        if original_dtype != x.dtype:
            x_normalized = x_normalized.to(original_dtype)
        return x_normalized

    def extra_repr(self):
        return f'dim={self.weight.shape[0]}, eps={self.eps}, do_cast={self.do_cast}'


def replace_layernorm_modules(module):
    for name, child in module.named_children():
        # 若是 torch 内置的 LayerNorm，则进行替换
        if isinstance(child, torch.nn.LayerNorm):
            # 从 child.normalized_shape 取出维度信息，这里假设是单维的情况
            if isinstance(child.normalized_shape, (tuple, list)):
                dim = child.normalized_shape[0]
            else:
                dim = child.normalized_shape
            new_ln = LayerNorm(dim, eps=child.eps, do_cast=False, affine=child.elementwise_affine)
            # 如原 LayerNorm 是仿射的，则复制其权重和偏置
            if child.elementwise_affine:
                new_ln = new_ln.to(child.weight.device, child.weight.dtype)
                new_ln.weight.data.copy_(child.weight.data)
                new_ln.bias.data.copy_(child.bias.data)
            module._modules[name] = new_ln
        else:
            replace_layernorm_modules(child)


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-Omni-3B"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="cpu")
    model = model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "https://huggingface.co/openbmb/MiniCPM-o-2_6/resolve/main/assets/input_examples/cxk_original.wav"},
                {"type": "text", "text": "将这段音频转为文字。"},
            ],
        },
    ]

    # set use audio in video
    USE_AUDIO_IN_VIDEO = True

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    print(inputs)
    input_features = inputs["input_features"]
    feature_attention_mask = inputs["feature_attention_mask"]

    with torch.no_grad():
        audio_embeds_noquant = model.thinker.get_audio_features(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )

        replace_layernorm_modules(model)
        audio_embeds = model.thinker.get_audio_features(
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )

    print(audio_embeds_noquant)
    print(f"{audio_embeds_noquant.shape = }")
    print(audio_embeds)
    print(f"{audio_embeds.shape = }")
    print(torch.allclose(audio_embeds_noquant, audio_embeds, atol=1e-5))

    diff = torch.abs(audio_embeds_noquant - audio_embeds)
    print(diff)
    print(f"{diff.max() = }, {diff.mean() = }, {diff.std() = }")

    cosine_sim = torch.cosine_similarity(audio_embeds_noquant, audio_embeds)
    print(cosine_sim)

    l2_dist = torch.norm(audio_embeds_noquant - audio_embeds, p=2)
    l2_sim = 1 / (1 + l2_dist)
    print(f"{l2_dist = }")
    print(f"{l2_sim = }")

    sqnr = 10 * torch.log10(torch.sum(audio_embeds_noquant ** 2) / torch.sum(diff ** 2))
    print(f"{sqnr = }")
    print(f"{sqnr.item() = } dB")
