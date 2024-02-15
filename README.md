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
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
