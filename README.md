このリポジトリは[第6回Brain(s)コンテスト by FUJIFILM AI Academy Brain(s)](https://fujifilmdatasciencechallnge.mystrikingly.com/)の解法です.

demo notebookにsolutionが記述されています.

### [🔥Demo Notebook(Google CoLab)](./solution.ipynb)

## requirements
```
albumentations              1.0.3
black                       21.7b0
flake8                      3.9.2
isort                       5.9.3
matplotlib                  3.4.3
mypy                        0.910
opencv-python               4.5.3.56
Pillow                      8.3.1
pytorch-widedeep            1.0.5
scikit-learn                0.24.2
segmentation-models-pytorch 0.2.0
timm                        0.4.12
torch                       1.8.1+cu111
torchaudio                  0.8.1
torchmetrics                0.5.0
torchvision                 0.9.1+cu111
```

## Checking format
```
sh check_format.sh DIR SCRIPT ...
```
e. g.
```
sh check_format.sh libs train.py train_seg.py
```

## Q1
in ```./solution.ipynb```

## Q2
### overview
![Q2](https://user-images.githubusercontent.com/38097069/132127332-f188dfce-8ac8-4876-a10d-dbeba284f799.png)

### training

- ***classification***
```
CUDA_VISIBLE_DEVICES=0 python3 train.py config/xxxx.yaml
```

- ***segmentation***
```
CUDA_VISIBLE_DEVICES=0 python3 train_seg.py config/xxxx.yaml
```

### inference
- ***classification***
```
CUDA_VISIBLE_DEVICES=0 python3 Q2_classification_inference.py config/xxxx.yaml --top-k 5
```

- ***segmentation***
```
CUDA_VISIBLE_DEVICES=0 python3 segmentation_inference.py config/xxxx.yaml --wall-type W --threshold 0.9
```

## Q3
### overview
![Q3](https://user-images.githubusercontent.com/38097069/131364158-337cfabf-4a88-44ad-a200-433932fd66c4.png)

### training
```
CUDA_VISIBLE_DEVICES=0 python3 train_seg.py config/xxxx.yaml
```

### inference
```
CUDA_VISIBLE_DEVICES=0 python3 segmentation_inference.py config/xxxx.yaml --threshold 0.5 --sub-pcon PCON_CONFIG_NAME --save-full-image
```

### zip
```
python3 pack.py config/xxxx.yaml
```
