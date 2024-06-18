# -*- coding: utf-8 -*-
import numpy
import pandas as pd
import matplotlib
import seaborn
import plotly
#opendatasets


dataset_url = 'https://github.com/JovianML/opendatasets/raw/master/data/house-prices-advanced-regression-techniques.zip'

from urllib.request import urlretrieve

urlretrieve(dataset_url, 'house-prices.zip')

from zipfile import ZipFile

with ZipFile('house-prices.zip') as f:
    f.extractall(path='house-prices')

import os

data_dir = 'house-prices'

os.listdir(data_dir)


import pandas as pd
pd.options.display.max_columns = 200
pd.options.display.max_rows = 200

train_csv_path = data_dir + '/train.csv'
train_csv_path


prices_df = pd.read_csv('house-prices/train.csv')

prices_df

prices_df.info()

n_rows = prices_df.shape[0]
n_rows

n_cols = prices_df.shape[1]
n_cols

print('The dataset contains {} rows and {} columns.'.format(n_rows, n_cols))


!pip install plotly matplotlib seaborn --quiet
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

prices_df.SalePrice.describe()

fig = px.scatter(prices_df,
                 x='SalePrice',
                 y='LotArea',
                 color='LotShape',
                 opacity=0.8,
                 title='SalePrice vs LotArea')
fig.update_traces(marker_size=5)
fig.show()

prices_df.HouseStyle.describe()

fig = px.scatter(prices_df,
                 x='SalePrice',
                 y='YearBuilt',
                 color='LotShape',
                 opacity=0.8,

                 title='SalePrice vs YearBuilt')
fig.update_traces(marker_size=5)
fig.show()


prices_df

# Identify the input columns (a list of column names)
input_cols = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']

# Identify the name of the target column (a single string, not a list)
target_col = 'SalePrice'

print(list(input_cols))

len(input_cols)

print(target_col)



inputs_df = prices_df[input_cols].copy()

targets = prices_df[target_col]

inputs_df

targets


prices_df.info()

import numpy as np

numeric_cols = inputs_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_cols = inputs_df.select_dtypes(include=['object']).columns.tolist()

print(list(numeric_cols))

print(list(categorical_cols))


missing_counts = inputs_df[numeric_cols].isna().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0]


from sklearn.impute import SimpleImputer


# 1. Create the imputer
imputer = SimpleImputer(strategy='mean')

# 2. Fit the imputer to the numeric colums
imputer.fit(prices_df[numeric_cols])

# 3. Transform and replace the numeric columns
inputs_df[numeric_cols] = imputer.transform(inputs_df[numeric_cols])


missing_counts = inputs_df[numeric_cols].isna().sum().sort_values(ascending=False)
missing_counts[missing_counts > 0] # should be an empty list


inputs_df[numeric_cols].describe().loc[['min', 'max']]


from sklearn.preprocessing import MinMaxScaler

# Create the scaler
scaler = MinMaxScaler()

# Fit the scaler to the numeric columns
scaler.fit(prices_df[numeric_cols])

# Transform and replace the numeric columns
inputs_df[numeric_cols] = scaler.transform(inputs_df[numeric_cols])


inputs_df[numeric_cols].describe().loc[['min', 'max']]


inputs_df[categorical_cols].nunique().sort_values(ascending=False)


from sklearn.preprocessing import OneHotEncoder

# 1. Create the encoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# 2. Fit the encoder to the categorical colums
encoder.fit(prices_df[categorical_cols])

# 3. Generate column names for each category
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
len(encoded_cols)

# 4. Transform and add new one-hot category columns
inputs_df[encoded_cols] = encoder.transform(inputs_df[categorical_cols])

inputs_df


from sklearn.model_selection import train_test_split

train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs_df[numeric_cols + encoded_cols],
                                                                        targets,
                                                                        test_size=0.25,
                                                                        random_state=42)

train_inputs

train_targets

val_inputs

val_targets

from sklearn.linear_model import Ridge

# Create the model
model = Ridge()

# Fit the model using inputs and targets
model.fit(train_inputs,train_targets)


from sklearn.metrics import mean_squared_error

train_preds =  model.predict(train_inputs)

train_preds

train_rmse = mean_squared_error(train_targets,train_preds,squared=False)

print('The RMSE loss for the training set is $ {}.'.format(train_rmse))

val_preds =  model.predict(val_inputs)

val_preds

val_rmse = mean_squared_error(val_targets,val_preds,squared=False)

print('The RMSE loss for the validation set is $ {}.'.format(val_rmse))


weights = model.coef_
len(weights)


weights_df = pd.DataFrame({
    'columns': train_inputs.columns,
    'weight': weights
}).sort_values('weight', ascending=False)

weights_df


from sklearn.tree import DecisionTreeRegressor

# Create the model
tree = DecisionTreeRegressor(random_state=42)

# Fit the model to the training data
tree.fit(train_inputs, train_targets)


from sklearn.metrics import mean_squared_error

tree_train_preds = tree.predict(train_inputs)
tree_train_rmse = mean_squared_error(train_targets,tree_train_preds,squared=False)
tree_val_preds = tree.predict(val_inputs)
tree_val_rmse = mean_squared_error(val_targets,tree_val_preds,squared=False)

print('Train RMSE: {}, Validation RMSE: {}'.format(tree_train_rmse, tree_val_rmse))


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
import seaborn as sns
sns.set_style('darkgrid')
# %matplotlib inline

plt.figure(figsize=(30,15))

# Visualize the tree graphically using plot_tree
x=plot_tree(tree, feature_names= train_inputs.columns, max_depth=3, filled=True);

# Visualize the tree textually using export_text
tree_text = export_text(tree)

# Display the first few lines
print(tree_text[:2000])

# Check feature importance
tree_importances =  tree.tree_.compute_feature_importances(tree)

tree_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': tree_importances
}).sort_values('importance', ascending=False)

tree_importance_df

plt.title('Decision Tree Feature Importance')
sns.barplot(data=tree_importance_df.head(10), x='importance', y='feature')


from sklearn.ensemble import RandomForestRegressor

# Create the model
rf1 = RandomForestRegressor(random_state=42)

# Fit the model
rf1.fit(train_inputs,train_targets)

rf1_train_preds = rf1.predict(train_inputs)
rf1_train_rmse = mean_squared_error(train_targets,rf1_train_preds,squared=False)
rf1_val_preds = rf1.predict(val_inputs)
rf1_val_rmse = mean_squared_error(val_targets,rf1_val_preds,squared=False)
print('Train RMSE: {}, Validation RMSE: {}'.format(rf1_train_rmse, rf1_val_rmse))


def test_params(**params):
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params).fit(train_inputs, train_targets)
    train_rmse = mean_squared_error(model.predict(train_inputs), train_targets, squared=False)
    val_rmse = mean_squared_error(model.predict(val_inputs), val_targets, squared=False)
    return train_rmse, val_rmse

test_params(n_estimators=20, max_depth=20)

test_params(n_estimators=50, max_depth=10, min_samples_leaf=4, max_features=0.4)

def test_param_and_plot(param_name, param_values):
    train_errors, val_errors = [], []
    for value in param_values:
        params = {param_name: value}
        train_rmse, val_rmse = test_params(**params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)
    plt.figure(figsize=(10,6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(['Training', 'Validation'])

test_param_and_plot('max_depth', [5, 10, 15, 20, 25, 30, 35])

test_param_and_plot('max_features' , [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85])

test_param_and_plot('n_estimators', [1, 5, 10, 15, 20, 25, 30, 35,40,45,50])

test_params(max_features='log2')

test_params(max_features=3)

test_params(max_leaf_nodes=2**5)

test_params(max_leaf_nodes=2**10)

test_params(min_samples_split=2, min_samples_leaf=2)

test_params(min_samples_split=6, min_samples_leaf=2)

test_param_and_plot('min_samples_split', [2, 3, 4, 5, 6, 7,8,9,10,12])

test_param_and_plot('min_samples_leaf', [1, 2, 3, 4, 5, 6, 7,8,9,10])


# Create the model with custom hyperparameters
#rf2 = RandomForestRegressor(n_estimators=50,max_leaf_nodes=2**20,max_features='log2',random_state=42)
#rf2 = RandomForestRegressor(n_estimators=50,max_depth=20,max_leaf_nodes=2**10,random_state=42)
#rf2 = RandomForestRegressor(n_estimators=50,min_samples_split=2, min_samples_leaf=2,random_state=42)
rf2=RandomForestRegressor(random_state=42)

# Train the model
rf2.fit(train_inputs, train_targets)

rf2_train_preds =rf2.predict(train_inputs)
rf2_train_rmse = mean_squared_error(train_targets,rf2_train_preds,squared=False)
rf2_val_preds = rf2.predict(val_inputs)
rf2_val_rmse = mean_squared_error(val_targets,rf2_val_preds,squared=False)

print('Train RMSE: {}, Validation RMSE: {}'.format(rf2_train_rmse, rf2_val_rmse))

"""Let's also view and plot the feature importances."""

rf2_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': rf2.feature_importances_
}).sort_values('importance', ascending=False)

rf2_importance_df

from matplotlib.pyplot import figure
figure(figsize=(8, 8))
sns.barplot(data=rf2_importance_df.head(10), x='importance', y='feature')



test_df = pd.read_csv('house-prices/test.csv')

test_df



# Apply the imputer and scaler transformations to the numeric columns
test_df[numeric_cols] = scaler.transform(imputer.transform(test_df[numeric_cols]))

# Apply the encoder transformation to the categorical columns
test_df[encoded_cols] = encoder.transform(test_df[categorical_cols])

test_inputs = test_df[numeric_cols + encoded_cols]


test_preds = rf2.predict(test_inputs)

submission_df = pd.read_csv('house-prices/sample_submission.csv')

submission_df


submission_df['SalePrice'] = test_preds




submission_df.to_csv('submission.csv', index=False)

from IPython.display import FileLink
FileLink('submission.csv') # Doesn't work on Colab, use the file browser instead to download the file.


import joblib

house_price_predictor = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}

joblib.dump(house_price_predictor, 'house_price_predictor.joblib')

