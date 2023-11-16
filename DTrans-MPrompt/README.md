## Requirements
- python==3.7.4
- pytorch==1.8.1
- [huggingface transformers](https://github.com/huggingface/transformers)
- numpy
- tqdm

## Overview
```
├── root
│   └── Entity-Detection
│       └── dataset
│           ├── conll2003_train.json
│           ├── conll2003_tag_to_id.json
│           ├── politics_train.json
│           ├── politics_dev.json
│           ├── politics_test.json
│           ├── politics_tag_to_id.json
│           └── ...
│       └── models
│           ├── __init__.py
│           └── modeling_span.py
│       └── utils
│           ├── __init__.py
│           ├── config.py
│           ├── data_utils.py
│           ├── eval.py
│           └── ...
│       └── ptms
│           └── ... (trained results, e.g., saved models, log file)
│       └── cached_models
│           └── ... (BERT-Base, which will be downloaded automatically)
│       └── run_script.py
│       └── run_script.sh
│   └── Type-Prediction
│       └── dataset
│           ├── conll2003_train.json
│           ├── conll2003_tag_to_id.json
│           ├── politics_train.json
│           ├── politics_dev.json
│           ├── politics_test.json
│           ├── politics_tag_to_id.json
│           └── ...
│       └── models
│           ├── __init__.py
│           └── modeling_type.py
│       └── utils
│           ├── __init__.py
│           ├── config.py
│           ├── data_utils.py
│           └── eval.py
│       └── ptms
│           └── ... (trained results, e.g., saved models, log file)
│       └── cached_models
│           └── ... (BERT-Base, which will be downloaded automatically)
│       └── run_script.py
│       └── run_script.sh
```

## How to run
### I. Entity Detection
```console
cd Entity-Detection/
```
#### 1. Training
```console
sh run_script.sh <GPU ID> <Target> True False <Source> Train
```
e.g., CoNLL2003 (Source) ---> politics (Target)
```console
sh run_script.sh 0 politics True False conll2003 Train
```
#### 2. Inference (Generate candidate entity spans)
```console
sh run_script.sh <GPU ID> <Target> False True <Source> <EVAL>
```
e.g., CoNLL2003 (Source) ---> politics (Target)
```console
sh run_script.sh 0 politics False True conll2003 dev
```
```console
sh run_script.sh 0 politics False True conll2003 test
```
#### 3. Copy the candidate entity spans into Type Prediction folder
```console
cp ptms/politics/dev_pred_spans.json ../Type-Prediction/dataset/
```
```console
cp ptms/politics/test_pred_spans.json ../Type-Prediction/dataset/
```

### II. Type Prediction
```console
cd ../Type-Prediction/
```
#### 1. Construct the inputs based on the candidate spans
```console
python dataset/combine.py --target politics --eval dev
```
```console
python dataset/combine.py --target politics --eval test
```
#### 2. Training
```console
sh run_script.sh <GPU ID> <Target> True False <Source> Train 1.0
```
e.g., CoNLL2003 (Source) ---> politics (Target)
```console
sh run_script.sh 0 politics True False conll2003 Train 1.0
```
#### 3. Inference
```console
sh run_script.sh 0 politics False True conll2003 test 0.6
```
