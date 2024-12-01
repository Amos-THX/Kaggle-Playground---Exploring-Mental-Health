# Kaggle-Playground---Exploring-Mental-Health
Kaggle Playground Competition Nov 2024

Notebook - https://www.kaggle.com/code/amosthx/kaggle-exploring-mental-health-nov-24/edit


## **1. Clean dataset**
- Check for na
- Ensure each column has proper unique values. Clean if necessary

## **2. Exploratory Data Analysis**
- Categorical Variables - do they have an impact on depression
- Correlation heatmap for numerical features

## **3. Preprocessing**
- Split datasets to working professionals vs students, train and test groups for training
- Perform OHE (not necessary for catboost) and MinMaxScaling

## **4. Modelling**
- Utilize RandomizedSearchCV to search for best parameters
- Train Catboost model for Students and Working Professional separately
- Generating dataset for submission


