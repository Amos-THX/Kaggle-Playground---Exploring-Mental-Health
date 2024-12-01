# %% [markdown]
# # 1. Kaggle Playground - Exploring Mental Health Data (Nov 2024)
# 
# ## **Step 1. Clean Dataset**
# - Check for na
# - Ensure each column has proper unique values. Clean if necessary
# 
# ## **Step 2. Exploratory Data Analysis**
# - Categorical Variables - do they have an impact on depression
# - Correlation heatmap for numerical features
# 
# ## **Step 3. Preprocessing**
# - Split datasets to working professionals vs students, train and test groups for training
# - Perform OHE (not necessary for catboost) and MinMaxScaling
# 
# ## **Step 4. Modelling**
# - Utilize RandomizedSearchCV to search for best parameters
# - Train Catboost model for Students and Working Professional separately
# - Generating dataset for submission

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:48.576489Z","iopub.execute_input":"2024-12-01T02:11:48.576888Z","iopub.status.idle":"2024-12-01T02:11:48.594705Z","shell.execute_reply.started":"2024-12-01T02:11:48.576853Z","shell.execute_reply":"2024-12-01T02:11:48.593686Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:48.597053Z","iopub.execute_input":"2024-12-01T02:11:48.597969Z","iopub.status.idle":"2024-12-01T02:11:48.626663Z","shell.execute_reply.started":"2024-12-01T02:11:48.597920Z","shell.execute_reply":"2024-12-01T02:11:48.625692Z"}}
import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import time
from scipy.stats import uniform

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
# import math

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:48.627868Z","iopub.execute_input":"2024-12-01T02:11:48.628273Z","iopub.status.idle":"2024-12-01T02:11:49.558745Z","shell.execute_reply.started":"2024-12-01T02:11:48.628229Z","shell.execute_reply":"2024-12-01T02:11:49.557534Z"}}
# Start the timer
start_time = time.time()

print(os.getcwd())

# Retrieve datasets
train = pd.read_csv('/kaggle/input/kaggle-playground-exploring-mental-health-data/train.csv')
test = pd.read_csv('/kaggle/input/kaggle-playground-exploring-mental-health-data/test.csv')

# %% [markdown]
# ## **Step 1. Clean Dataset**
# - Check for na
# - Ensure each column has proper unique values. Clean if necessary

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:49.560992Z","iopub.execute_input":"2024-12-01T02:11:49.561330Z","iopub.status.idle":"2024-12-01T02:11:49.644050Z","shell.execute_reply.started":"2024-12-01T02:11:49.561298Z","shell.execute_reply":"2024-12-01T02:11:49.642983Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# 1. Check na
check_na = train.isna().sum()

check_na

# Dataset contains 2 sets of data for working professional / student

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:49.645209Z","iopub.execute_input":"2024-12-01T02:11:49.645528Z","iopub.status.idle":"2024-12-01T02:11:49.661878Z","shell.execute_reply.started":"2024-12-01T02:11:49.645497Z","shell.execute_reply":"2024-12-01T02:11:49.660806Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Ensure each column has proper unique values. Clean if necessary

# Name - ignore

# Gender
print(train['Gender'].unique())
# Gender is either Male/Female. No Cleaning required
# Perform OHE


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:49.663283Z","iopub.execute_input":"2024-12-01T02:11:49.664066Z","iopub.status.idle":"2024-12-01T02:11:49.909331Z","shell.execute_reply.started":"2024-12-01T02:11:49.664017Z","shell.execute_reply":"2024-12-01T02:11:49.908296Z"},"_kg_hide-input":false,"_kg_hide-output":true}

# Age
print(train['Age'].unique())  # All whole numbers between a range
# Check distribution for histogram output - Equal distribution from 20-60
plt.hist(train['Age'])
plt.show()
# Relatively equal distribution - Perform MinMaxScaler

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:49.910573Z","iopub.execute_input":"2024-12-01T02:11:49.910940Z","iopub.status.idle":"2024-12-01T02:11:50.197002Z","shell.execute_reply.started":"2024-12-01T02:11:49.910908Z","shell.execute_reply":"2024-12-01T02:11:50.195968Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# City
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# print(len(train['City'].unique())) #98 different cities, check for outliers
# Check value counts
print(train['City'].value_counts().sort_values(ascending=True))


# print(test['City'].value_counts().sort_values(ascending=True)  )

# Based on train dataset, change City to Null if value count <=7. then do SimpleImputer
def clean_city(df):
    '''
    If city appears less than 10 times, classify as Others

    Parameters:
    df - train/test dataset

    Returns:
    df - dataset with City cleaned.

    '''

    train_city_counts = df['City'].value_counts() <= 10
    exclude_city_list = train_city_counts[train_city_counts == True].index.tolist()
    df['City'] = df['City'].apply(lambda x: x if x not in exclude_city_list else 'Others')
    df['City'] = df['City'].replace({None: np.nan})
    return df


# Clean train and test data set, replace outliers with Null
train = clean_city(train)
test = clean_city(test)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:50.198279Z","iopub.execute_input":"2024-12-01T02:11:50.198603Z","iopub.status.idle":"2024-12-01T02:11:50.212590Z","shell.execute_reply.started":"2024-12-01T02:11:50.198571Z","shell.execute_reply":"2024-12-01T02:11:50.211412Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Working Professional or Student
print(train['Working Professional or Student'].unique())  # - No cleaning req
# Perform OHE

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:50.213860Z","iopub.execute_input":"2024-12-01T02:11:50.214163Z","iopub.status.idle":"2024-12-01T02:11:54.610998Z","shell.execute_reply.started":"2024-12-01T02:11:50.214134Z","shell.execute_reply":"2024-12-01T02:11:54.609902Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Profession
print(train['Profession'].value_counts())


# Try with best effort to replace
def clean_profession(df):
    '''
    Create student part time flag with a different profession.

    Try on best case basis to group professions like student, consultant, doctor into 1 group.

    For professions with less than 10 entries, classify as 'Others'. Classify null as Others

    Parameters:
    df - train/test dataset

    Returns:
    df - dataset with Profession cleaned
    '''

    df['Profession'] = df['Profession'].astype(str)

    df['student_parttime_flag'] = df.apply(
        lambda x: 1 if x['Profession'] != 'nan' and x['Working Professional or Student'] == 'Student' else 0, axis=1)

    df['Profession'] = df.apply(
        lambda x: 'Student' if x['Working Professional or Student'] == 'Student' and x['Profession'] == None else x[
            'Profession'], axis=1)
    df['Profession'] = df['Profession'].apply(lambda x: 'Student' if x in ('Academic', 'Student', 'PhD', 'MBA') else x)

    df['Profession'] = df['Profession'].apply(lambda x: 'Doctor' if 'Doctor' in x else x)
    df['Profession'] = df['Profession'].apply(lambda x: 'Consultant' if 'Consultant' in x else x)

    train_prof_counts = df['Profession'].value_counts() <= 10
    exclude_prof_list = train_prof_counts[train_prof_counts == True].index.tolist()
    df['Profession'] = df['Profession'].apply(lambda x: x if x not in exclude_prof_list else 'Others')

    df['Profession'] = df['Profession'].apply(lambda x: 'Others' if x == 'nan' else x)

    return df


train = clean_profession(train)
test = clean_profession(test)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:54.614530Z","iopub.execute_input":"2024-12-01T02:11:54.614908Z","iopub.status.idle":"2024-12-01T02:11:54.627783Z","shell.execute_reply.started":"2024-12-01T02:11:54.614867Z","shell.execute_reply":"2024-12-01T02:11:54.626623Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Academic Pressure
print(train['Academic Pressure'].value_counts())  # ~70% of dataset is null.
# Work Pressure
print(train['Work Pressure'].value_counts())  # ~70% of dataset is null.

# Academic Pressure for students, work pressure for working professionals. fill with median

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:54.629173Z","iopub.execute_input":"2024-12-01T02:11:54.629523Z","iopub.status.idle":"2024-12-01T02:11:54.839798Z","shell.execute_reply.started":"2024-12-01T02:11:54.629479Z","shell.execute_reply":"2024-12-01T02:11:54.838674Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# CGPA - missing for ~70% of dataset - only for students
print(train['CGPA'].value_counts().head(5))

# Check distribution for histogram output
plt.hist(train['CGPA'])
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:54.841065Z","iopub.execute_input":"2024-12-01T02:11:54.841448Z","iopub.status.idle":"2024-12-01T02:11:57.020214Z","shell.execute_reply.started":"2024-12-01T02:11:54.841413Z","shell.execute_reply":"2024-12-01T02:11:57.019306Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Study Satisfaction - Work Satisfaction - Combine datasets
# print( train['Study Satisfaction'].value_counts() )

def satisfacton(df):
    '''
    Combine job / study satisfaction into 1 field.

    Parameters:
    df - train/test dataset

    Returns:
    Df with satisfaction column

    '''
    df['Satisfaction'] = df.apply(
        lambda x: x['Job Satisfaction'] if x['Job Satisfaction'] > 0 else x['Study Satisfaction'], axis=1)
    df['Satisfaction'] = df['Satisfaction'].fillna(0)

    return df


train = satisfacton(train)
test = satisfacton(test)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:57.021969Z","iopub.execute_input":"2024-12-01T02:11:57.022400Z","iopub.status.idle":"2024-12-01T02:11:57.093620Z","shell.execute_reply.started":"2024-12-01T02:11:57.022352Z","shell.execute_reply":"2024-12-01T02:11:57.092589Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Sleep Duration - mostly in 4 categories
print(train['Sleep Duration'].value_counts())


# Slot all others into 7-8 hours for ease of use

def sleep_hours(df):
    '''
    For ease of cleaning, slot other categories into the standard 7-8 hour range.

    Parameters:
    df - train/test dataset

    Returns:
    df with sleep duration cleaned
    '''

    df['Sleep Duration'] = df['Sleep Duration'].apply(
        lambda x: x if x in ('Less than 5 hours', '7-8 hours', 'More than 8 hours', '5-6 hours') else '7-8 hours')
    return df


train = sleep_hours(train)
test = sleep_hours(test)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:57.094774Z","iopub.execute_input":"2024-12-01T02:11:57.095085Z","iopub.status.idle":"2024-12-01T02:11:57.192293Z","shell.execute_reply.started":"2024-12-01T02:11:57.095055Z","shell.execute_reply":"2024-12-01T02:11:57.191193Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Dietary Habits - Main Categories are Moderate, Unhealthy, Healthy
print(train['Dietary Habits'].value_counts())


# Clean up all categories
def dietary_habits(text):
    if text == 'Healthy' or text == 'Yes' or text == 'More Healthy' or text == '1.0' or text == '1':
        return 'Healthy'
    elif text == 'Unhealthy' or text == 'No' or text == 'No Healthy' or text == 'Less Healthy' or text == '3' or text == 'Less than Healthy':
        return 'Unhealthy'
    elif text == 'Moderate' or text == '2':
        return 'Moderate'
    else:
        return 'Moderate'  # Set the others to Moderate


def dietary(df):
    '''
    Clean up dietary habits on best case basis.

    Parameters:
    df - train/test dataset

    Returns:
    df with Dietary Habits cleaned
    '''

    df['Dietary Habits'] = df['Dietary Habits'].apply(dietary_habits)

    return df


train = dietary(train)
test = dietary(test)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:57.193775Z","iopub.execute_input":"2024-12-01T02:11:57.194082Z","iopub.status.idle":"2024-12-01T02:11:57.540190Z","shell.execute_reply.started":"2024-12-01T02:11:57.194052Z","shell.execute_reply":"2024-12-01T02:11:57.539265Z"},"_kg_hide-input":false,"_kg_hide-output":true}
print(train['Degree'].value_counts())


def degree(df):
    '''
    Remove all small naming conventions and slot into "Others". Fill null as "Others".

    Parameters:
    df - train/test dataset

    Returns:
    df with degree cleaned
    '''

    train_degree_count = df['Degree'].value_counts() < 10
    exclude_degree_list = train_degree_count[train_degree_count == True].index.tolist()
    df['Degree'] = df['Degree'].apply(lambda x: x if x not in exclude_degree_list else 'Others')

    df['Degree'] = df['Degree'].fillna('Others')

    return df


train = degree(train)
test = degree(test)

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:57.541381Z","iopub.execute_input":"2024-12-01T02:11:57.541700Z","iopub.status.idle":"2024-12-01T02:11:57.558821Z","shell.execute_reply.started":"2024-12-01T02:11:57.541669Z","shell.execute_reply":"2024-12-01T02:11:57.557671Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Check Label - Have you ever had suicidal thoughts ?
print(train['Have you ever had suicidal thoughts ?'].value_counts())

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:57.560306Z","iopub.execute_input":"2024-12-01T02:11:57.560746Z","iopub.status.idle":"2024-12-01T02:11:57.797954Z","shell.execute_reply.started":"2024-12-01T02:11:57.560682Z","shell.execute_reply":"2024-12-01T02:11:57.796961Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Work/Study Hours
print(train['Work/Study Hours'].value_counts())

# Check distribution - Roughly equal distribution from left to right --> use Min Max Scaler
plt.bar(train['Work/Study Hours'].value_counts().index, train['Work/Study Hours'].value_counts().values)
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:57.799499Z","iopub.execute_input":"2024-12-01T02:11:57.799830Z","iopub.status.idle":"2024-12-01T02:11:57.816480Z","shell.execute_reply.started":"2024-12-01T02:11:57.799798Z","shell.execute_reply":"2024-12-01T02:11:57.815261Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Financial Stress
print(train['Financial Stress'].value_counts())


def financial_stress(df):
    '''
    Fill null Financial Stress as most frequent. 

    Parameters:
    df 

    Returns:
    df 

    '''
    df['Financial Stress'] = df['Financial Stress'].fillna(train['Financial Stress'].value_counts().max())
    return df


train = financial_stress(train)
test = financial_stress(test)
# Use Min Max Scaler, equal distribution from 1-5

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:57.818030Z","iopub.execute_input":"2024-12-01T02:11:57.818577Z","iopub.status.idle":"2024-12-01T02:11:57.849525Z","shell.execute_reply.started":"2024-12-01T02:11:57.818528Z","shell.execute_reply":"2024-12-01T02:11:57.848367Z"},"_kg_hide-input":false,"_kg_hide-output":true}
# Family History of Mental Illness - Only Yes No
print(train['Family History of Mental Illness'].value_counts())

# %% [markdown]
# ## **Step 2. Exploratory Data Analysis**
# - Categorical Variables - do they have an impact on depression
# - Correlation heatmap for numerical features

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T05:52:27.446100Z","iopub.execute_input":"2024-12-01T05:52:27.447036Z","iopub.status.idle":"2024-12-01T05:52:30.153088Z","shell.execute_reply.started":"2024-12-01T05:52:27.446991Z","shell.execute_reply":"2024-12-01T05:52:30.151949Z"}}
for col in ['Gender', 'Profession', 'City', 'Sleep Duration', 'Dietary Habits', 'Degree',
            'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
    df = train[train['Working Professional or Student'] == 'Working Professional'].groupby(col).agg(
        pct=('Depression', 'mean'))
    plt.figure(figsize=(12, 8))
    plt.bar(df.index, df['pct'])
    plt.title('Working Professional - ' + col)
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T05:51:31.632178Z","iopub.execute_input":"2024-12-01T05:51:31.632596Z","iopub.status.idle":"2024-12-01T05:51:33.723069Z","shell.execute_reply.started":"2024-12-01T05:51:31.632560Z","shell.execute_reply":"2024-12-01T05:51:33.721937Z"}}
for col in ['Gender', 'City', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
            'Family History of Mental Illness']:
    df = train[train['Working Professional or Student'] == 'Student'].groupby(col).agg(pct=('Depression', 'mean'))
    plt.figure(figsize=(12, 8))
    plt.bar(df.index, df['pct'])
    plt.title('Student - ' + col)
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:20:03.408286Z","iopub.execute_input":"2024-12-01T02:20:03.408659Z","iopub.status.idle":"2024-12-01T02:20:03.995529Z","shell.execute_reply.started":"2024-12-01T02:20:03.408629Z","shell.execute_reply":"2024-12-01T02:20:03.994401Z"}}
# Plotting correlation heatmap for numeric values
plt.figure(figsize=(12, 6))
dataplot = sns.heatmap(train[train['Working Professional or Student'] == 'Working Professional'].drop(
    columns=['Academic Pressure', 'Study Satisfaction', 'student_parttime_flag']).corr(numeric_only=True),
                       cmap="YlGnBu", annot=True)
plt.title("Working Professional Correlation")
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:20:36.141424Z","iopub.execute_input":"2024-12-01T02:20:36.141831Z","iopub.status.idle":"2024-12-01T02:20:36.667958Z","shell.execute_reply.started":"2024-12-01T02:20:36.141789Z","shell.execute_reply":"2024-12-01T02:20:36.666780Z"}}
# Plotting correlation heatmap for numeric values
plt.figure(figsize=(12, 6))
dataplot = sns.heatmap(train[train['Working Professional or Student'] == 'Student'].drop(
    columns=['Work Pressure', 'Job Satisfaction', 'Satisfaction']).corr(numeric_only=True), cmap="YlGnBu", annot=True)
plt.title("Student Correlation")
plt.show()


# %% [markdown]
# ## **Step 3. Preprocessing**
# - Split datasets to working professionals vs students, train and test groups for training
# - Perform OHE (not necessary for catboost) and MinMaxScaling

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:58.572594Z","iopub.execute_input":"2024-12-01T02:11:58.573589Z","iopub.status.idle":"2024-12-01T02:11:58.702108Z","shell.execute_reply.started":"2024-12-01T02:11:58.573539Z","shell.execute_reply":"2024-12-01T02:11:58.701140Z"}}
def split_data(df):
    """performs data split for students vs working professional, train and test dataset

    Parameters:
    - df - split dataset

    Returns:
    student_x_train - student training dataset
    student_y_train - student dataset with label
    working_x_train - working professional training dataset
    working_y_test - working professional dataset with label
    """

    student_df = df[df['Working Professional or Student'] == 'Student']
    student_x = student_df[['id', 'Age', 'Academic Pressure', 'Satisfaction', 'Work/Study Hours', 'Financial Stress',
                            'student_parttime_flag',
                            'Gender', 'City', 'Sleep Duration',
                            'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?',
                            'Family History of Mental Illness', 'CGPA']]

    working_df = df[df['Working Professional or Student'] == 'Working Professional']

    working_x = working_df[
        ['id', 'Age', 'Work Pressure', 'Satisfaction', 'Work/Study Hours', 'Financial Stress', 'CGPA',
         'Gender', 'City', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree',
         'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']]

    # For test datasets, simply return training datasets for prediction
    try:
        student_y = student_df[['Depression']]
        working_y = working_df[['Depression']]

        print("y datasets generated")
    except:
        return student_x, working_x  # For final test dataset
        print('test data without labels')

    return student_x, student_y, working_x, working_y


# Extract train test datasets
student_x, student_y, working_x, working_y = split_data(train)

final_X_student, final_X_working = split_data(test)

print(student_x.head(1))


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:58.703341Z","iopub.execute_input":"2024-12-01T02:11:58.703649Z","iopub.status.idle":"2024-12-01T02:11:59.359829Z","shell.execute_reply.started":"2024-12-01T02:11:58.703620Z","shell.execute_reply":"2024-12-01T02:11:59.358542Z"}}
def get_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111, stratify=y)

    return X_train, X_test, y_train, y_test


student_X_train, student_X_test, student_y_train, student_y_test = get_train_test(student_x, student_y)
working_X_train, working_X_test, working_y_train, working_y_test = get_train_test(working_x, working_y)


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:59.361707Z","iopub.execute_input":"2024-12-01T02:11:59.362223Z","iopub.status.idle":"2024-12-01T02:11:59.373747Z","shell.execute_reply.started":"2024-12-01T02:11:59.362177Z","shell.execute_reply":"2024-12-01T02:11:59.372452Z"}}
def one_hot_encoder(train_df, test_df, final_df, ohe_cols):
    """
    Perform one-hot encoding on specified columns and return transformed DataFrames.

    Parameters:
    - train_df: pd.DataFrame, training data
    - test_df: pd.DataFrame, testing data
    - final_df: final dataset for submission
    - ohe_cols: list, columns to one-hot encode

    Returns:
    - train_df: pd.DataFrame, transformed training data
    - test_df: pd.DataFrame, transformed testing data
    """

    for col, strategy in [('Financial Stress', 'median')]:
        try:
            imputer = SimpleImputer(strategy=strategy)
            train_df[col] = imputer.fit_transform(train_df[[col]])
            test_df[col] = imputer.transform(test_df[[col]])
            final_df[col] = imputer.transform(final_df[[col]])
        except:
            pass

    # Check for missing columns in train and test DataFrames
    missing_train_cols = [col for col in ohe_cols if col not in train_df.columns]
    missing_test_cols = [col for col in ohe_cols if col not in test_df.columns]

    if missing_train_cols or missing_test_cols:
        print(f"Warning: Missing columns in train: {missing_train_cols}, test: {missing_test_cols}")
        # Add missing columns with NaN
        for col in missing_train_cols:
            train_df[col] = None
        for col in missing_test_cols:
            test_df[col] = None

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit and transform the columns to encode
    encoded_data = encoder.fit_transform(train_df[ohe_cols])
    # Create a DataFrame for the encoded columns
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(ohe_cols))

    # Drop original columns and concatenate the encoded ones
    train_df = pd.concat([train_df.drop(columns=ohe_cols).reset_index(drop=True), encoded_df.reset_index(drop=True)],
                         axis=1)

    encoded_data = encoder.transform(test_df[ohe_cols])
    # Create a DataFrame for the encoded columns
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(ohe_cols))
    # Drop original columns and concatenate the encoded ones
    test_df = pd.concat([test_df.drop(columns=ohe_cols).reset_index(drop=True), encoded_df.reset_index(drop=True)],
                        axis=1)

    encoded_data = encoder.transform(final_df[ohe_cols])
    # Create a DataFrame for the encoded columns
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(ohe_cols))
    # Drop original columns and concatenate the encoded ones
    final_df = pd.concat([final_df.drop(columns=ohe_cols).reset_index(drop=True), encoded_df.reset_index(drop=True)],
                         axis=1)

    return train_df, test_df, final_df


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:59.375255Z","iopub.execute_input":"2024-12-01T02:11:59.375701Z","iopub.status.idle":"2024-12-01T02:11:59.390027Z","shell.execute_reply.started":"2024-12-01T02:11:59.375657Z","shell.execute_reply":"2024-12-01T02:11:59.388932Z"}}
def min_max_scaler(train_df, test_df, final_df, min_max_cols):
    """
    Perform one-hot encoding on specified columns and return transformed DataFrames.

    Parameters:
    - train_df: pd.DataFrame, training data
    - test_df: pd.DataFrame, testing data
    - final_df: final dataset for submission
    - min_max_cols: list, columns to apply Min-Max scaling (not implemented in this function)

    Returns:
    - train_df: pd.DataFrame, transformed training data
    - test_df: pd.DataFrame, transformed testing data
    - final_df: pd.DataFrame, final submission dataframe
    """

    for col, strategy in [('CGPA', 'median'), ('Degree', 'most_frequent')]:
        try:
            imputer = SimpleImputer(strategy=strategy)
            train_df[col] = imputer.fit_transform(train_df[[col]])
            test_df[col] = imputer.transform(test_df[[col]])
            final_df[col] = imputer.transform(final_df[[col]])
        except:
            pass

    # Perform MinMaxScaler
    for col in min_max_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_df[col] = scaler.fit_transform(train_df[[col]])
        test_df[col] = scaler.transform(test_df[[col]])
        final_df[col] = scaler.transform(final_df[[col]])

    return train_df, test_df, final_df


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:59.391528Z","iopub.execute_input":"2024-12-01T02:11:59.391946Z","iopub.status.idle":"2024-12-01T02:11:59.511472Z","shell.execute_reply.started":"2024-12-01T02:11:59.391902Z","shell.execute_reply":"2024-12-01T02:11:59.510363Z"}}
# Perform all preprocessing steps - Catboost used so OHE is not required for categorical variables.
# student_X_train, student_X_test,final_X_student = one_hot_encoder(student_X_train,student_X_test,final_X_student, ['Gender', 'City','Sleep Duration', 'Dietary Habits', 'Degree','Have you ever had suicidal thoughts ?','Family History of Mental Illness'])
# working_X_train, working_X_test,final_X_working = one_hot_encoder(working_X_train,working_X_test,final_X_working, ['Gender', 'City', 'Profession','Sleep Duration', 'Dietary Habits', 'Degree','Have you ever had suicidal thoughts ?','Family History of Mental Illness'])

student_X_train, student_X_test, final_X_student = min_max_scaler(student_X_train, student_X_test, final_X_student,
                                                                  ['CGPA', 'Age', 'Academic Pressure', 'Satisfaction',
                                                                   'Work/Study Hours', 'Financial Stress'])
working_X_train, working_X_test, final_X_working = min_max_scaler(working_X_train, working_X_test, final_X_working,
                                                                  ['CGPA', 'Age', 'Work Pressure', 'Satisfaction',
                                                                   'Work/Study Hours', 'Financial Stress'])


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:59.513171Z","iopub.execute_input":"2024-12-01T02:11:59.513575Z","iopub.status.idle":"2024-12-01T02:11:59.556409Z","shell.execute_reply.started":"2024-12-01T02:11:59.513528Z","shell.execute_reply":"2024-12-01T02:11:59.555415Z"}}
def drop_id(df):
    df = df.drop(columns=['id'])
    return df


student_X_train, student_X_test, final_X_student2 = drop_id(student_X_train), drop_id(student_X_test), drop_id(
    final_X_student)
working_X_train, working_X_test, final_X_working2 = drop_id(working_X_train), drop_id(working_X_test), drop_id(
    final_X_working)


# %% [markdown]
# ## **Step 4. Modelling**
# - Utilize RandomizedSearchCV to search for best parameters
# - Train Catboost model for Students and Working Professional separately
# - Generating dataset for submission

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:59.561830Z","iopub.execute_input":"2024-12-01T02:11:59.562187Z","iopub.status.idle":"2024-12-01T02:11:59.569941Z","shell.execute_reply.started":"2024-12-01T02:11:59.562153Z","shell.execute_reply":"2024-12-01T02:11:59.568876Z"}}

def randomized_search_cv(X_train, X_test, y_train, y_test):
    '''
    Perform randomized search cross validation on training dataset. Check using accuracy (metric for competition).

    Parameters:
    X_train - training dataset with features
    X_test - test dataset with features
    y_train - training dataset label
    y_test - test dataset label

    Returns:
    Prints out best parameters and accuracy for test dataset
    Returns model with best estimators
    '''

    param_dist = {
        # 'num_leaves': np.arange(20, 150, 10),
        'max_depth': np.arange(1, 16, 2),  # Catboost
        'learning_rate': np.logspace(-3, -1, 10),  # Catboost
        'n_estimators': np.arange(50, 500, 50),  # Catboost
        # 'min_child_samples': np.arange(10, 100, 10),
        # 'l2_leaf_reg': [1, 3, 5, 7, 9],
        # 'subsample': [0.6, 0.8, 1.0],
        # 'random_strength': [1, 2, 5, 10]
        # 'C': uniform(0.01, 10),  # Regularization strength (inverse of regularization)
        # 'penalty': ['l1', 'l2'],  # Regularization type
        # 'solver': ['liblinear']  # Solver compatible with the chosen penalties
    }

    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=CatBoostClassifier(cat_features=['Gender', 'City', 'Sleep Duration', 'Dietary Habits', 'Degree',
                                                   'Have you ever had suicidal thoughts ?',
                                                   'Family History of Mental Illness']),
        param_distributions=param_dist,
        n_iter=100,  # Number of random combinations to try
        scoring='accuracy',  # Adjust this for your problem (e.g., 'roc_auc', 'f1')
        cv=10,  # 10-fold cross-validation
        verbose=2,
        random_state=111,
        n_jobs=-1  # Use all available cores
    )

    # Fit the random search model
    random_search.fit(X_train, y_train)

    # Best parameters
    print("Best Parameters:", random_search.best_params_)

    # Predict using the best model
    best_model = random_search.best_estimator_
    y_pred_final = best_model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred_final)
    print(f"Test Accuracy: {accuracy:.4f}")

    return best_model


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T02:11:59.571102Z","iopub.execute_input":"2024-12-01T02:11:59.571409Z","iopub.status.idle":"2024-12-01T02:11:59.585519Z","shell.execute_reply.started":"2024-12-01T02:11:59.571379Z","shell.execute_reply":"2024-12-01T02:11:59.584433Z"}}
# Perform randomized search cv on both student and working professionals separately for CatboostClassifier
# best_model_student = randomized_search_cv(student_X_train, student_X_test,  student_y_train, student_y_test)
# best_model_working = randomized_search_cv(working_X_train, working_X_test,  working_y_train, working_y_test)

# %% [markdown]
# ### Optimal results after tuning
# - Working / Catboost {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.05994842503189409} Accuracy = 0.9618
# 
# - Student / Catboost {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05994842503189409} Accuracy = 0.8496

# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T05:53:27.871329Z","iopub.execute_input":"2024-12-01T05:53:27.872147Z","iopub.status.idle":"2024-12-01T05:53:27.877668Z","shell.execute_reply.started":"2024-12-01T05:53:27.872108Z","shell.execute_reply":"2024-12-01T05:53:27.876624Z"}}
def model_accuracy(model, X_train, X_test, y_train, y_test):
    '''
    Input model type, parameters, and datasets to retrieve accuracy score for train test split

    Parameters:
    model - key in model class and specific parameters
    X_train - training dataset features
    X_test - test dataset features
    y_train - training dataset label
    y_test - test dataset label

    Output:
    Print accuracy score
    Returns back model for use.
    '''

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Evaluate the ensemble model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    return model


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T05:54:10.711650Z","iopub.execute_input":"2024-12-01T05:54:10.712047Z","iopub.status.idle":"2024-12-01T05:54:41.400379Z","shell.execute_reply.started":"2024-12-01T05:54:10.712011Z","shell.execute_reply":"2024-12-01T05:54:41.399199Z"}}
working_model = model_accuracy(CatBoostClassifier(n_estimators=400, max_depth=7, learning_rate=0.05994842503189409,
                                                  cat_features=['Gender', 'City', 'Profession', 'Sleep Duration',
                                                                'Dietary Habits', 'Degree',
                                                                'Have you ever had suicidal thoughts ?',
                                                                'Family History of Mental Illness']), working_X_train,
                               working_X_test, working_y_train, working_y_test)
student_model = model_accuracy(CatBoostClassifier(n_estimators=200, max_depth=5, learning_rate=0.05994842503189409,
                                                  cat_features=['Gender', 'City', 'Sleep Duration', 'Dietary Habits',
                                                                'Degree', 'Have you ever had suicidal thoughts ?',
                                                                'Family History of Mental Illness']), student_X_train,
                               student_X_test, student_y_train, student_y_test)


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T05:55:17.559568Z","iopub.execute_input":"2024-12-01T05:55:17.559973Z","iopub.status.idle":"2024-12-01T05:55:17.566048Z","shell.execute_reply.started":"2024-12-01T05:55:17.559938Z","shell.execute_reply":"2024-12-01T05:55:17.564967Z"}}
def output_test_data(model, X_train, X_test, y_train):
    '''
    Input model type, parameters, and datasets to retrieve accuracy score for train test split

    Parameters:
    model - key in model class and specific parameters
    X_train - training dataset features
    X_test - final test dataset without labels
    y_train - training dataset label


    Output:
    y_prediction - list of predictions for final output to submit

    '''
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test.drop(columns=['id']))
    X_test['prediction'] = y_pred

    return X_test[['id', 'prediction']]


# %% [code] {"execution":{"iopub.status.busy":"2024-12-01T05:56:09.431330Z","iopub.execute_input":"2024-12-01T05:56:09.432669Z","iopub.status.idle":"2024-12-01T05:56:43.557833Z","shell.execute_reply.started":"2024-12-01T05:56:09.432611Z","shell.execute_reply":"2024-12-01T05:56:43.556916Z"}}
final_student = output_test_data(student_model, student_X_train, final_X_student, student_y_train)
final_working = output_test_data(working_model, working_X_train, final_X_working, working_y_train)
#
submission = pd.concat([final_working, final_student], axis=0)

submission.to_csv('amosthx_submission.csv', index=False)

# %% [code]
