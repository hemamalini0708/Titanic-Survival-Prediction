import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import logging
import sys
from logg_file import phase_1
logger = phase_1("feature_selection")
from sklearn.feature_selection import VarianceThreshold
quansi_con = VarianceThreshold(threshold=0.1)


def constant_tech(train_data,test_data):
    try:
        quansi_con.fit(train_data)
        sol = train_data.columns[~quansi_con.get_support()]
        logger.info(f"Columns with 0.1 variances are : {sol}")
        return sol
    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from line no : {er_lineno.tb_lineno} Issue : {er_msg}")
