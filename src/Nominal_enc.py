import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import sklearn
import logging
from logg_file import phase_1
logger = phase_1("Nominal_enc")
from sklearn.preprocessing import OneHotEncoder
one_hot_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # ✅ Fixed line


# to convert columns  categorical to numerical
# It works on Equality Base columns only
def cat_num_one_hot(train_cat, test_cat):
    try:
        # Fit encoder on actual column values from train_cat
        one_hot_enc.fit(train_cat[['Sex', 'Embarked']])
        sol = one_hot_enc.transform(train_cat[['Sex', 'Embarked']])

        for i, name in enumerate(one_hot_enc.get_feature_names_out(['Sex', 'Embarked'])):
            train_cat[name] = sol[:, i]

        sol_ = one_hot_enc.transform(test_cat[['Sex', 'Embarked']])
        for i, name in enumerate(one_hot_enc.get_feature_names_out(['Sex', 'Embarked'])):
            test_cat[name] = sol_[:, i]

        # Drop original categorical columns
        train_cat = train_cat.drop(['Sex', 'Embarked'], axis=1)
        test_cat = test_cat.drop(['Sex', 'Embarked'], axis=1)

        logger.info("Categorical columns converted into numerical successfully")
        return train_cat, test_cat

    except Exception as e:
        er_type, er_msg, er_tb = sys.exc_info()
        logger.error(f"Error from line no : {er_tb.tb_lineno} Issue : {er_msg}")



