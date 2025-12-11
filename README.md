# FMSA-SC: A Fine-grained Multimodal Sentiment Analysis Dataset based on Stock Comment Videos



 Previous Sentiment Analysis (SA) studies have demonstrated that exploring sentiment cues from multiple synchronized modalities can effectively improve the SA results. Unfortunately, until now there is no publicly available dataset for multimodal SA of the stock market. Existing datasets for stock market SA only provide textual stock comments, which usually contain words with ambiguous sentiments or even sarcasm words expressing opposite sentiments of literal meaning. To address this issue, we introduce a Fine-grained Multimodal Sentiment Analysis dataset built upon 1, 247 Stock Comment videos, called FMSA-SC. It provides both multimodal sentiment annotations for the videos and unimodal sentiment annotations for the textual, visual, and acoustic modalities of the videos. In addition, FMSASC also provides fine-grained annotations that align text at the phrase level with visual and acoustic modalities. Furthermore, we present a new fine-grained multimodal multi-task framework as the baseline for multimodal SA on the FMSA-SC.

![](./img/1710939837773.png)


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
