# DVC commands

These are the DVC commands I used to produce the stages in the `dvc.yaml` file.

## Data preparation stage

```bash
dvc run -n prepare \
-p prepare.train_size,prepare.test_size,prepare.random_state \
-d src/prepare.py -d data/raw/train.csv -d data/raw/test.csv
-o data/processed
python src/prepare.py
```

## Model training stage

```bash
dvc run -n train \
-p train.random_state,train.algorithm \
-d src/train.py -d data/processed/X_train.csv -d data/processed/y_train.csv \
-o models \
python src/train.py
```

## Model evaluation stage

```bash
dvc run -n evaluate \
-d models/iowa_model.pkl -d src/evaluate.py -d data/processed/X_valid.csv -d data/processed/y_valid.csv \
-M metrics/scores.json \
python src/evaluate.py
```