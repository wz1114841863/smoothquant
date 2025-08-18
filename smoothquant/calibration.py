import torch
import torch.nn as nn

from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm


def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    """计算模型的激活缩放因子"""
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        """统计张量的最大值"""
        hidden_dim = tensor.shape[-1]  # 获取隐藏层维度
        tensor = tensor.view(-1, hidden_dim).abs().detach()  # 展平张量, 取绝对值并分离计算图
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()  # 计算每列(沿隐藏维度)的最大值
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)  # 更新最大值
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        """作为钩子函数, 在模型前向传播时收集输入数据"""
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)  # 执行模型推理(触发钩子函数收集激活统计信息)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def get_static_decoder_layer_scales(
    model,
    tokenizer,
    dataset_path,
    num_samples=512,
    seq_len=512,
):
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        """对输入(x)和输出(y)的绝对值的全局最大值进行动态更新, 保留所有样本中的最大值"""
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item()
            )
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item()
            )

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)
        # 实时计算所有层输入的平均尺度, 用于监控数据分布.
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()

    decoder_layer_scales = []
    for idx in range(model.config.num_hidden_layers):
        scale_dict = {}
        # 所有统计的最大值除以127(INT8对称量化范围[-127, 127]),直接得到量化Scale参数
        scale_dict["attn_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["input"] / 127
        )
        scale_dict["q_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.q_proj"]["output"] / 127
        )
        scale_dict["k_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.k_proj"]["output"] / 127
        )
        scale_dict["v_output_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.v_proj"]["output"] / 127
        )
        scale_dict["out_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.self_attn.out_proj"]["input"] / 127
        )
        scale_dict["fc1_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.fc1"]["input"] / 127
        )
        scale_dict["fc2_input_scale"] = (
            act_dict[f"model.decoder.layers.{idx}.fc2"]["input"] / 127
        )
        decoder_layer_scales.append(scale_dict)

    return decoder_layer_scales, act_dict
