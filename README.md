# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
[[paper](https://arxiv.org/abs/2211.10438)] [[slides](assets/SmoothQuant.pdf)][[video](https://youtu.be/U0yvqjdMfr0)]

![intuition](figures/intuition.png)

源地址：https://github.com/mit-han-lab/smoothquant

## Installation

```bash
uv venv --python 3.8
source ./.venv/bin/activate

uv pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
uv pip install transformers==4.36.0 accelerate datasets zstandard
uv pip install setuptools

python setup.py install

git clone --recurse-submodules https://github.com/Guangxuan-Xiao/torch-int.git
cd ./torch-int
uv pip install -r requirements.txt
source environment.sh
bash build_cutlass.sh
python setup.py install
python tests/test_linear_modules.py
```

## Usage

```bash
# 以opt为例实现伪量化, 对比结果
1. python ./examples/smoothquant_opt_demo.py

# 获取act-scales, 使用校准数据集中各通道激活的最大值, 用于计算缩放因子.
2. python ./examples/generate_act_scales.py

# 导出INT8模型
python ./examples/export_int8_model.py

# 推理INT8模型, 验证结果
# 报错, bias的维度不匹配, 未能成功解决
python ./examples/smoothquant_opt_real_int8_demo.py

```
