import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# Path of the input data folder
input_folder_path = Path('./input')

# Path of the files to read
train_path = input_folder_path / 'train.csv'
test_path = input_folder_path / 'test.csv'

# Read dataset from csv file
train_data = pd.read_csv(train_path, index_col='Id')
test_data = pd.read_csv(test_path, index_col='Id')


# ================ #
# DATA PREPARATION #
# ================ #

# Remove rows with missing target
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# Separate target from predictors
y = train_data.SalePrice

# Create a DataFrame called `X` holding the predictive features.
X_full = train_data.drop(['SalePrice'], axis=1)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = test_data.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8,
                                                      test_size=0.2,
                                                      random_state=0)

# Handle Missing Values with Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; I put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
X_train = imputed_X_train
X_valid = imputed_X_valid

# Path of the output data folder
Path('prepared').mkdir(exist_ok=True)
prepared_folder_path = Path('prepared')

X_train_path = prepared_folder_path / 'X_train.csv'
y_train_path = prepared_folder_path / 'y_train.csv'
X_valid_path = prepared_folder_path / 'X_valid.csv'
y_valid_path = prepared_folder_path / 'y_valid.csv'

X_train.to_csv(X_train_path)
print("Writing file {} to disk.".format(X_train_path))

y_train.to_csv(y_train_path)
print("Writing file {} to disk.".format(y_train_path))

X_valid.to_csv(X_valid_path)
print("Writing file {} to disk.".format(X_valid_path))

y_valid.to_csv(y_valid_path)
print("Writing file {} to disk.".format(y_valid_path))

