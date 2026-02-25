import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import logging
from sklearn.model_selection import train_test_split
import sys
from logg_file import phase_1
logger = phase_1("median_mode")

def missing_data_med_mode(X_train, X_test):
    try:
        logger.info("Handling missing values with median/mode...")

        # Calculate imputation values from training data only (prevent data leakage)
        age_median = X_train['Age'].median()
        embarked_mode = X_train['Embarked'].mode()[0]

        # Apply to training data
        X_train['Age'] = X_train['Age'].fillna(age_median)
        X_train['Embarked'] = X_train['Embarked'].fillna(embarked_mode)

        # Apply same values to test data (using training data statistics)
        X_test['Age'] = X_test['Age'].fillna(age_median)
        X_test['Embarked'] = X_test['Embarked'].fillna(embarked_mode)

        logger.info("Missing values filled successfully")
        return X_train, X_test

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from line no: {er_lineno.tb_lineno} Issue: {er_msg}")
        return X_train, X_test
