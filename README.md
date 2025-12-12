# MGMIN-FSA: A Multi-Granularity Multimodal Interaction Network for Sentiment Analysis of Financial Review Videos



Automated sentiment analysis of extensive financial review videos provides investors with a precise understanding of market sentiment. However, compared to videos in other domains, financial review videos contain extensive financial jargon and exhibit more nuanced sentiment changes in facial expressions and vocal tones. Most traditional models focus solely on either coarse-grained or fine-grained multimodal sentiment expression. This limitation renders them ineffective in handling videos with subtle sentiment changes, restricting their applicability to analysing sentiment in financial review videos. This paper introduces the Multi-Granularity Multimodal Interaction Network (MGMIN-FSA) for analysing sentiment in financial review videos. First, the Fine-grained Multimodal Interaction Network (FMN) is presented, incorporating Kolmogorov-Arnold Network (KAN) and attention mechanisms to extract crucial phrase-level features across modalities, producing fine-grained sentiment expressions. Additionally, KAN-Attention within FMN improves the model's generalization capability. Second, the Coarse-grained Feature Decoupling Network (CFDN) is introduced, establishing anchors at both modal and sentiment feature levels to improve differentiation between positive and negative samples. Finally, MGMIN-FSA integrates fine- and coarse-grained representations for sentiment analysis using downstream classifiers. MGMIN-FSA obtains the best performance on the FMSA-SC and CH-SIMS datasets among all compared methods. Furthermore, experimental results demonstrate that KAN-Attention can maintain high performance despite insufficient data. Our codes can be found at https://github.com/junzhangNLP/MGMIN-FSA.

![](./img.png)
![](./FIG-re/fig3.pdf)
## Project Structure
```python3
├── dataset/                   # Dataset directory
│   ├── CH-SIMS/               # CH-SIMS dataset
│   ├── fasmr/                 # fasmr datasets
│   ├── mosei_test.pkl         # MOSEI dataset
│   ├── mosei_train.pkl        # MOSEI dataset
│   └── mosei_valid.pkl        # MOSEI dataset
├── model/                     # Model definitions
│   ├── __pycache__/          # Python cache files
│   └── model.py              # Main model implementation
├── tools/                     # Tools and utility functions
│   ├── __pycache__/
│   ├── CH_dataloader.py      # CH dataset loader
│   ├── Config.py             # Configuration class
│   ├── data_loader.py        # Generic data loader
│   ├── DataLoaderCMUSDK.py   # CMUSDK dataset loader
│   ├── DataLoaderUniversal.py # Universal data loader
│   ├── function.py           # Utility functions
│   ├── kan.py                # KAN network implementation
│   └── Utils.py              # Utility functions
└── main.py                   # Main program entry point
```

## 1. Dataset Download Instructions

CH-SIMS Dataset：https://github.com/thuiar/ch-sims-v2

fasmr Dataset：https://github.com/sunlitsong/FMSA-SC-dataset

MOSEI：https://github.com/kiva12138/CubeMLP

## 2. Path Configuration for Local Environment
Step 1: Locate Configuration Files

check main.py for any paths and config

Step 2: Modify Dataset Paths and Pretrained Model Paths

check any .py for any paths

## 3. Installation and Setup

Normally the pip install command will handle the python dependency packages. (requirements.txt)

## 4. Run

We have provided the start code in the "main.py".

use: ```python3 python main.py ``` to run.
