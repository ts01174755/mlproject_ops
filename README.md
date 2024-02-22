kaggle_home_credit
==============================

# 安裝 Git bash

1. 下載 Git bash:

    https://git-scm.com/downloads

2. 更換環境: 
    
    a. 重啟 VSCode
    
    b. crtl + shift + p
    
    c. Terminal: select Default Profile

    d. 選擇 Git Bash

# 使用 python 自帶 venv

``` 建構環境: python -m venv venv```

啟用: interpreter 選擇虛擬環境

# 安裝 CUDA & pytorch

## 儲存環境

``` [interpreter_path]/pip.exe freeze > requirements.txt ```

## 安裝環境

``` [interpreter_path]/pip.exe install -r requirements.txt ```

## 選擇要使用的pytorch版本以及對應的CUDA

- 對應: https://blog.csdn.net/qq_41946216/article/details/129476095

- CUDA-12.1 下載: https://developer.nvidia.com/cuda-12-1-0-download-archive

- pytorch-11.7 下載: https://pytorch.org/get-started/locally/

    ```$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile 文件，包含像是 `make data` 或 `make train` 的命令
    ├── README.md          <- 開發者使用此項目的頂層自述檔案。
    ├── data
    │   ├── external       <- 第三方來源的數據。
    │   ├── interim        <- 已轉換的中間數據。
    │   ├── processed      <- 模型化用的最終、標準數據集。
    │   └── raw            <- 原始的、不可變的數據傾印。
    │
    ├── docs               <- 默認的 Sphinx 項目；詳情見 sphinx-doc.org
    │
    ├── models             <- 訓練並序列化的模型、模型預測或模型摘要
    │
    ├── notebooks          <- Jupyter 筆記本。命名規則是一個數字（用於排序）、
    │                         創建者的首字母，以及一個短的 `-` 分隔的描述，例如
    │                         `1.0-jqp-initial-data-exploration`。
    │
    ├── references         <- 數據字典、手冊以及所有其他解釋性材料。
    │
    ├── reports            <- 生成的分析報告，如 HTML、PDF、LaTeX 等。
    │   └── figures        <- 用於報告的生成圖表和圖形
    │
    ├── requirements.txt   <- 重現分析環境所需的文件，例如
    │                         通過 `pip freeze > requirements.txt` 生成
    │
    ├── setup.py           <- 使項目可通過 pip 安裝（pip install -e .）以便可以導入 src
    ├── src                <- 用於此項目的源代碼。
    │   ├── __init__.py    <- 使 src 成為一個 Python 模塊
    │   │
    │   ├── data           <- 下載或生成數據的腳本
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- 將原始數據轉換為模型化特徵的腳本
    │   │   └── build_features.py
    │   │
    │   ├── models         <- 訓練模型並使用訓練好的模型進行預測的腳本
    │   │   │                 預測
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- 創建探索性和結果導向的可視化的腳本
    │       └── visualize.py
    │
    └── tox.ini            <- 帶有運行 tox 設置的 tox 文件；詳情見 tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
