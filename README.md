# Recurrent_Meta_Learning_for_CTR
An implement of the ACM MM 22 paper: Recurrent Meta-Learning against Generalized Cold-Start Problem in CTR Prediction.

## Environments
* **Python** 3.6.8
* **tensorflow** 1.14

## Data
Unzip the preprocessed dataset avazu.7z, bookcrossing.7z, ml-1m.7z to ./data/avazu, ./data/bookcrossing, ./data/ml-1m.

## Training
1. Modify configs in `config.py` (copy the parameters from `config [dataset] [subset].py`)
2. Run the script:
```shell
CUDA_VISIBLE_DEVICES=0  python main.py --model our --dataset ml-1m --start 0 --end 1
```
Note: `--start 0 --end 1` for WC subset and  `--start 1 --end 2` for CW subset.

## Evaluation
1. Modify configs in `config.py` (copy the parameters from `config [dataset] [subset].py`)
2. Run the script:
```shell
CUDA_VISIBLE_DEVICES=0  python main_test.py --model our --dataset ml-1m --start 0 --end 1
```
