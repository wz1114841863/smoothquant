import torch
import argparse
import os

from pathlib import Path

from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers import AutoTokenizer

from smoothquant.opt import Int8OPTForCausalLM
from smoothquant.smooth import smooth_lm

from smoothquant.calibration import get_static_decoder_layer_scales


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default='facebook/opt-125m')
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--act-scales", type=str,
                        default='act_scales/opt-125m.pt')
    parser.add_argument("--output-path", type=str, default='int8_models')
    parser.add_argument('--dataset-path', type=str, default='/mnt/d/Cache/pile_datasets/val.jsonl.zst',
                        help='location of the calibration dataset, we use the validation set of the Pile dataset')
    parser.add_argument('--export-FT', default=False, action="store_true")
    args = parser.parse_args()

    model = OPTForCausalLM.from_pretrained(
        args.model_name, device_map="auto", torch_dtype=torch.float16)
    act_scales = torch.load(args.act_scales)
    smooth_lm(model, act_scales, 0.5)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # print(f"finished smoothing {args.model_name} model")

    if not os.path.exists(args.dataset_path):
        print(f'Cannot find the dataset at {args.dataset_path}')
        print('Please download the Pile dataset and put the validation set at the path')
        print('You can download the validation dataset of the Pile at https://mystic.the-eye.eu/public/AI/pile/val.jsonl.zst')
        raise FileNotFoundError

    # 重新跑 512 条文本,记录 平滑后每一层 Linear 的 整层最大绝对值
    # 然后除以 127 得到 7 个 per-tensor scale(decoder_layer_scales)
    decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(model,
                                                                       tokenizer,
                                                                       args.dataset_path,
                                                                       num_samples=args.num_samples,
                                                                       seq_len=args.seq_len)
    output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant.pt")
    if args.export_FT:
        # 只保存 平滑后的 float16 模型 和 scale 文件,
        model.save_pretrained(output_path)
        print(f"Saved smoothed model at {output_path}")

        output_path = Path(args.output_path) / (Path(args.model_name).name + "-smoothquant-scales.pt")
        torch.save(raw_scales, output_path)
        print(f"Saved scaling factors at {output_path}")
    else:
        # 已量化成 INT8 的权重 
        int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
        int8_model.save_pretrained(output_path)
        print(f"Saved int8 model at {output_path}")
