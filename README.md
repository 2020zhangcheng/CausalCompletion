# Robust Point Cloud Completion via Causal-Guided Invariant Representation Learning

# Introduction
This repository is the code for my paper: Robust Point Cloud Completion via Causal-Guided Invariant Representation Learning

Our framework is model-agnostic and can be seamlessly integrated into existing completion networks. It consists of a DBAD module to suppress observation-dependent features and a CGC module to encourage consistent completion predictions under diverse observation conditions. Qualitative and quantitative evaluations on several benchmark datasets demonstrate that our approach achieves strong performance.

# Installation
```
git clone https://github.com/2020zhangcheng/CausalCompletion.git
cd CausalCompletion
conda create --name CausalCompletion python=3.11.0
conda activate CausalCompletion
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
sh extensions/install.sh
```

# Training and Testing
1. Download datasets

    [PCN dataset](https://gateway.infinitescript.com/s/ShapeNetCompletion)


    [ShapeNet55/34](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view)

    [Real Sensors Dataset]([[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/076097900274447cb3bd/?dl=1) or [[Google Drive]](https://drive.google.com/file/d/1OzQg1-_GefA8NOVa8h8Y4oPoouUHerVI/view?usp=sharing). )
