# GALK-VT-Net
GALK-VT Net: A Medical Segmentation Framework Based on Global Attention Large Kernel and Vision-Text Prompt Learning

## 🛠️ 环境准备与数据配置

### 1. 数据集下载
为了复现本项目的结果，你需要下载以下三个公开数据集：
* **AMOS22**: [Multi-Modality Abdominal Multi-Organ Segmentation Challenge](https://amos22.grand-challenge.org/)
* **BTCV**: [Beyond the Cranial Vault Segmentation Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/6035)
* **KiTS19**: [Kidney Tumor Segmentation Challenge](https://kits19.grand-challenge.org/)

### 2. 数据目录结构
下载解压后，请将数据按照以下结构放入项目根目录的 `data` 文件夹下：

```text
GALK-VT-Net/
├── data/
│   ├── AMOS22/
│   │   ├── imagesTr/
│   │   └── labelsTr/
│   ├── BTCV/
│   │   ├── RawData/
│   │   └── ...
│   └── KiTS19/
│       ├── case_00000/
│       └── ...
├── clip/
├── main.py
└── README.md
