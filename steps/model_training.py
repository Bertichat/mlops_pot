import logging 
import pandas as pd

from zenml import step


# Import create an ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# To create a checkpoint and save the best model
from tensorflow.keras.callbacks import ModelCheckpoint

# To load the model
from tensorflow.keras.models import load_model

# To check the metrics of the model
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import numpy as np
    
@step
def train_model(X_train: np.ndarray, 
                X_test: np.ndarray, 
                y_train: np.ndarray, 
                y_test: np.ndarray) -> Sequential:
    try:
        # Crete a Sequential Object
        model = Sequential()
        # Add first layer with 100 neurons to the sequental object
        model.add(Dense(100,input_shape=(40,),activation='relu'))
        # Add second layer with 200 neurons to the sequental object
        model.add(Dense(100,activation='relu'))
        # Add third later with 100 neurons to the sequental object
        model.add(Dense(100,activation='relu'))

        # Output layer With 10 neurons as it has 10 classes
        model.add(Dense(10,activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                optimizer='adam')
        
        # Set the number of epochs for training
        num_epochs = 100
        # Set the batch size for training
        batch_size = 32

        # Fit the model
        model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=num_epochs,batch_size=batch_size,verbose=1)
        return model
    except Exception as e:
        logging.error(f"Error Evaluating: {e}")
        raise e