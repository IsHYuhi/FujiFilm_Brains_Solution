このリポジトリは[第6回Brain(s)コンテスト by FUJIFILM AI Academy Brain(s)](https://fujifilmdatasciencechallnge.mystrikingly.com/)の解法です.

demo notebookにsolutionが記述されています.

### [🔥Demo Notebook(Google CoLab)](./solution.ipynb)

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
### training
```
CUDA_VISIBLE_DEVICES=0 python3 train_seg.py config/xxxx.yaml
```

### inference
```
CUDA_VISIBLE_DEVICES=0 python3 segmentation_inference.py config/xxxx.yaml --threshold 0.5
```

### zip
```
python3 pack.py config/xxxx.yaml
```
