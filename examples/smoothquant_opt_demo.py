import torch
from transformers.models.opt.modeling_opt import (
    OPTAttention,
    OPTDecoderLayer,
    OPTForCausalLM,
)
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear, quantize_opt
from datasets import load_dataset


class Evaluator:
    """预测下一个单词的准确率的简单评估"""
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc


tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")
dataset = load_dataset("lambada", split="validation[:1000]")
evaluator = Evaluator(dataset, tokenizer, "cuda")

# # FP16 Model Accuracy: 1.3b-acc:0.721
# model_fp16 = OPTForCausalLM.from_pretrained(
#     "facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto"
# )
# acc_fp16 = evaluator.evaluate(model_fp16)
# print(f"Original model (fp16) accuracy: {acc_fp16}")

# # Naive W8A8 Quantization: 1.3b-acc:0.689
# model_w8a8 = quantize_opt(model_fp16)
# print(model_w8a8)
# acc_w8a8 = evaluator.evaluate(model_w8a8)
# print(f"Naive W8A8 quantized model accuracy: {acc_w8a8}")

# smoothquant quantization W8A8 1.3b-acc:0.706
model = OPTForCausalLM.from_pretrained(
    "facebook/opt-1.3b", torch_dtype=torch.float16, device_map="auto"
)
act_scales = torch.load("./act_scales/opt-1.3b.pt")
smooth_lm(model, act_scales, 0.5)
model_smoothquant_w8a8 = quantize_opt(model)
print(model_smoothquant_w8a8)
acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
print(f"SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8}")
