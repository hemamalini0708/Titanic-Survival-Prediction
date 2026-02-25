import pickle
import os
import sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from logg_file import phase_1

logger = phase_1("final_testing")

def testing_():
    try:
        model_ = pickle.load(open("Titanic_model.pkl", 'rb'))
        logger.info(f"Model loaded successfully: {type(model_)}")

        temp = np.random.random((1, 7))  # 7 features

        prediction = model_.predict(temp)[0]
        logger.info(f"Prediction Result: {prediction}")

        if prediction == 0:
            result = 'Did Not Survive'
        else:
            result = 'Survived'

        logger.info(f"Final Output: {result}")
        return result

    except Exception as e:
        er_type, er_msg, er_lineno = sys.exc_info()
        logger.error(f"Error from Line no : {er_lineno.tb_lineno} Issue : {er_msg}")
        return "Prediction failed due to an error."
