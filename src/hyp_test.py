import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import logging
from sklearn.model_selection import train_test_split
import sys
from logg_file import phase_1
logger = phase_1("hyp_test")
from scipy.stats import pearsonr

def hypothesis(X_train, Y_train, X_test, Y_test):
    try:
        waste_cols = []
        p_values = {}

        # Perform Pearson correlation for each column
        for i in X_train.columns:
            corr, p = pearsonr(X_train[i], Y_train)
            p_values[i] = p
            logger.info(f"Column: {i} | Correlation: {corr:.4f} | p-value: {p:.4f}")

            if p > 0.05:
                waste_cols.append(i)

        # Drop statistically insignificant columns
        X_train = X_train.drop(waste_cols, axis=1)
        X_test = X_test.drop(waste_cols, axis=1)

        logger.info(f"Dropped columns: {waste_cols}")
        return X_train, X_test

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")



