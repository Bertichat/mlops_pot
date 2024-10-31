import logging 
import pandas as pd

from zenml import step
from tensorflow.keras.models import Sequential
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix


@step
def evaluate_model(model: Sequential, 
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> pd.DataFrame:
    try:
        y_pred = model.predict(X_test)
        y_pred = [np.argmax(i) for i in y_pred]
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        print(report)
        return df_report
    except Exception as e:
        logging.error(f"Error Evaluating: {e}")
        raise e