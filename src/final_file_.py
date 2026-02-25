import numpy as np
import pandas as pd
import sklearn
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from logg_file import phase_1
import pickle
import matplotlib.pyplot as plt
logger = phase_1("final_file")

def f_m(final_X_train,final_y_train,final_X_test,final_y_test):
    try:
        # logistic regression
        NB_reg = GaussianNB()
        NB_reg.fit(final_X_train, final_y_train)
        NB_pred = NB_reg.predict(final_X_test)
        with open('Titanic_model.pkl','wb') as f_:
            pickle.dump(NB_reg,f_)
        logger.info("final Model saved Successfully")

    except Exception as e:
        er_type,er_msg,er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")