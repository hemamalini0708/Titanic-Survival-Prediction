import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sklearn
import sys
import logging
from logg_file import phase_1
logger = phase_1("models")
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


def NB(X_train,Y_train,X_test,Y_test):
    try:
        NB_reg = GaussianNB()
        NB_reg.fit(X_train,Y_train)
        logger.info(f"NB Accuray : {accuracy_score(Y_test, NB_reg.predict(X_test))}")
        logger.info(f"NB Confusion Matrix : {confusion_matrix(Y_test, NB_reg.predict(X_test))}")
        logger.info(f"NB Classification Report : {classification_report(Y_test, NB_reg.predict(X_test))}")
    except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def LR(X_train,Y_train,X_test,Y_test):
    try:
        LR_reg = LogisticRegression()
        LR_reg .fit(X_train,Y_train)
        logger.info(f"LR Accuray : {accuracy_score(Y_test, LR_reg .predict(X_test))}")
        logger.info(f"LR Confusion Matrix : {confusion_matrix(Y_test, LR_reg .predict(X_test))}")
        logger.info(f"LR Classification Report : {classification_report(Y_test, LR_reg .predict(X_test))}")
    except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def DT(X_train,Y_train,X_test,Y_test):
    try:
        DT_reg = DecisionTreeClassifier(criterion='entropy')
        DT_reg.fit(X_train,Y_train)
        logger.info(f"DT Accuray : {accuracy_score(Y_test,  DT_reg.predict(X_test))}")
        logger.info(f"DT Confusion Matrix : {confusion_matrix(Y_test,  DT_reg.predict(X_test))}")
        logger.info(f"DT Classification Report : {classification_report(Y_test,  DT_reg.predict(X_test))}")
    except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")

def multi_models(X_train,Y_train,X_test,Y_test):
    try:
        logger.info(f"---------Navi_Bayes-----------")
        NB(X_train, Y_train, X_test, Y_test)
        logger.info(f"------Logistic Regression------")
        LR(X_train, Y_train, X_test, Y_test)
        logger.info(f"--------Decision Tree---------")
        DT(X_train, Y_train, X_test, Y_test)
    except Exception as e:
            er_type, er_msg, er_lineno = sys.exc_info()
            logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")