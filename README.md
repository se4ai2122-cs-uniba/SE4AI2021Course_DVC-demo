# DVC demo for the SE4AI 2021-22 course

This is a demo ML project used to show the main features of [DVC](https://dvc.org) in the 2021 edition of the Software Engineering for AI-enabled Systems course (University of Bari, Italy - Dept. of Computer Science).

The scripts used in this project are freely inspired by the [Kaggle](https://www.kaggle.com) Tutorial ["Intermediate Machine Learning"](https://www.kaggle.com/learn/intermediate-machine-learning). Accordingly, the example uses data from the [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/c/home-data-for-ml-course).


## Import raw data

As a first step, we imported raw data using the [`dvc import`](https://dvc.org/doc/command-reference/import) command:

```bash
dvc import https://github.com/collab-uniba/Software-Solutions-for-Reproducible-ML-Experiments input/home-data-for-ml-course/train.csv -o data/raw

dvc import https://github.com/collab-uniba/Software-Solutions-for-Reproducible-ML-Experiments input/home-data-for-ml-course/test.csv -o data/raw
```

Observe that, although available in Kaggle, these data files were taken from another public GitHub repository, [Software Solutions for Reproducible ML Experiments](https://github.com/collab-uniba/Software-Solutions-for-Reproducible-ML-Experiments), to demonstrate this capability of DVC.

## Setup a Python environment

Then, we created a Python (virtual) environment and installed the requirements for this project, which are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```


## Run the ML pipeline stages via DVC

Finally, we executed the following three DVC `run` commands, corresponding to the three stages of this simple ML pipeline (data preparation, model training, and model evaluation).

### Data preparation stage

```bash
dvc run -n prepare \
-p prepare.train_size,prepare.test_size,prepare.random_state \
-d src/prepare.py -d data/raw/train.csv -d data/raw/test.csv \
-o data/processed/X_train.csv -o data/processed/X_valid.csv \
-o data/processed/y_train.csv -o data/processed/y_valid.csv \
python src/prepare.py
```

### Model training stage

```bash
dvc run -n train \
-p train.random_state,train.algorithm \
-d src/train.py -d data/processed/X_train.csv -d data/processed/y_train.csv \
-o models/iowa_model.pkl \
python src/train.py
```

### Model evaluation stage

```bash
dvc run -n evaluate \
-d models/iowa_model.pkl -d src/evaluate.py -d data/processed/X_valid.csv -d data/processed/y_valid.csv \
-M metrics/scores.json \
python src/evaluate.py
```


## Reproducing the whole pipeline

The details about each stage are automatically stored by DVC in the `dvc.yaml` file. 

To reproduce the entire pipeline, it is sufficient to run:

```bash
dvc repro
```

---

The scripts from this repo are also available as [a GitHub Gist](https://gist.github.com/louieQ/55c9845fa131a5defd359999afeba0fa).
