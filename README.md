# DVC demo for the SE4AI 2021-22 course

The code in this repo is supposed to be shared with the students via [this GitHub Gist](https://gist.github.com/louieQ/55c9845fa131a5defd359999afeba0fa):
it contains the python scripts from this repo plus a couple of config files
(`requirements.txt` and `params.yaml`).

## DVC `run` commands

These are the DVC `run` commands that I used to generate the stages in the `dvc.yaml` file.

### Data preparation stage

```bash
dvc run -n prepare \
-p prepare.train_size,prepare.test_size,prepare.random_state \
-d src/prepare.py -d data/raw/train.csv -d data/raw/test.csv
-o data/processed
python src/prepare.py
```

### Model training stage

```bash
dvc run -n train \
-p train.random_state,train.algorithm \
-d src/train.py -d data/processed/X_train.csv -d data/processed/y_train.csv \
-o models \
python src/train.py
```

### Model evaluation stage

```bash
dvc run -n evaluate \
-d models/iowa_model.pkl -d src/evaluate.py -d data/processed/X_valid.csv -d data/processed/y_valid.csv \
-M metrics/scores.json \
python src/evaluate.py
```