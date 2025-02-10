# Imports:

import pandas as pd
import numpy as np

from sklearn.linear_model  import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# Defining a function for categorical variables visualization:

def cat_var(df, cols):
    '''
    Return: a Pandas dataframe object with the following columns:
        - "categorical_variable" => every categorical variable include as an input parameter (string).
        - "number_of_possible_values" => the amount of unique values that can take a given categorical variable (integer).
        - "values" => a list with the posible unique values for every categorical variable (list).

    Input parameters:
        - df -> Pandas dataframe object: a dataframe with categorical variables.
        - cols -> list object: a list with the name (string) of every categorical variable to analyse.
    '''
    cat_list = []
    for col in cols:
        cat = df[col].unique()
        cat_num = len(cat)
        cat_dict = {"categorical_variable":col,
                    "number_of_possible_values":cat_num,
                    "values":cat}
        cat_list.append(cat_dict)
    df = pd.DataFrame(cat_list).sort_values(by="number_of_possible_values", ascending=False)
    return df.reset_index(drop=True)


# Function for one-hot-encoding:

def one_hot_encod(df, cat_lst):
    """
    This function performs one-hot encoding on categorical variables in a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing both categorical and non-categorical variables.
    cat_lst (list): A list of column names in the DataFrame that are categorical and need to be one-hot encoded.

    Returns:
    pandas.DataFrame: A DataFrame where the specified categorical columns have been replaced with their one-hot encoded equivalents,
                      while the non-categorical columns remain unchanged.
    """
    cat_var_df = df[cat_lst]
    non_cat_var_df = df.drop(cat_lst, axis=1)
    cat_var_df_encoded = pd.get_dummies(cat_var_df, drop_first=True, dtype=int)
    df_encoded = pd.concat([non_cat_var_df, cat_var_df_encoded], axis=1)
    return df_encoded


# Function for error evaluation:

def cross_val(model, features, target):
    """
    This function evaluates the performance of a given model using cross-validation and computes the root mean squared error (RMSE) for each fold.

    Parameters:
    model (sklearn model): A machine learning model object (e.g., regression or classifier) to be evaluated using cross-validation.
    features (pandas.DataFrame or numpy.array): The feature set used for model evaluation.
    target (pandas.Series or numpy.array): The target values corresponding to the feature set.

    Returns:
    None: The function prints the cross-validation scores for each fold and the mean RMSE across all folds.
    """
    scores = cross_val_score(model, 
                            features, 
                            target, 
                            scoring='neg_root_mean_squared_error', 
                            cv=5,
                            n_jobs=-1)
    print(f'Cross val. scores: {scores}', '\n')
    print(f'Mean of scores: {np.mean(-scores)}', '\n')


# Function to save predictions:

def save_pred(df_test, predictions, file_name):
    """
    This function saves the predictions made on a test dataset to a CSV file.

    Parameters:
    df_test (pandas.DataFrame): The test DataFrame containing an 'id' column, used to identify each data point.
    predictions (numpy.array or pandas.Series): The predicted values (e.g., predicted prices) corresponding to the test dataset.
    file_name (str): The name of the CSV file where the predictions will be saved (without file extension).

    Returns:
    str: A message indicating that the file has been saved successfully.
    """
    path = '../predictions/' + file_name + '.csv'
    id_col = df_test['id'].to_numpy()
    pred_df = pd.DataFrame({'id': id_col, 'price': predictions})
    pred_df.to_csv(path, index=False)
    return 'file saved succesfully'