import logging 
import pandas as pd
import librosa
import numpy as np

from zenml import step
from sklearn.model_selection import train_test_split
from typing import Union, Tuple
from typing_extensions import Annotated


SR = 22050
def amp(x):
    _ = librosa.stft(x)
    return _

def amp_db(x):
    k = librosa.amplitude_to_db(abs(librosa.stft(x)))
    return k

def spec_cent(x):
    return librosa.feature.spectral_centroid(y=x, sr=SR)[0]

def mel_spectro(x):
    return librosa.feature.melspectrogram(y=x, sr=SR, n_fft=2048, hop_length=512, n_mels=10)[0]

def mfcc(x):
    mfcc =  librosa.feature.mfcc(y=x,sr=SR, n_mfcc=40)
    mfcc = np.mean(mfcc.T,axis=0)
    return mfcc

class DataCleaning():
    def __init__(self, data):
        self.df = data

    def clean_data(self) -> pd.DataFrame:
        #Getting the amplitude to decibel
        #flattening for pipe dimension
        #getting throux X,y faster may solve it also, else reshape.
        self.df["amplitude_db"] = self.df["voice_array"].apply(amp_db)
        self.df["amplitude_db_shape"] = self.df["amplitude_db"].apply(lambda x: x.shape)
        self.df["amplitude_db"] = self.df["amplitude_db"].apply(lambda x: x.flatten())

        self.df["mfcc"] = self.df["voice_array"].apply(mfcc)
        # self.df["mfcc_shape"] = self.df["mfcc_shape"].apply(lambda x: x.shape)
        # self.df["mfcc"] = self.df["mfcc"].apply(lambda x: x.flatten())

        self.df["spectral_centroid"] = self.df["voice_array"].apply(spec_cent)

        self.df["mel_spectro"] = self.df["voice_array"].apply(mel_spectro)

        return(self.df)
    
class DataDividing():
    def __init__(self, data):
        self.df = data

    def divide_data(self) -> Union[np.ndarray, np.ndarray]:
        #divide the data in labels and features
        #array values types as float 32 or crash the pipe.
        X = np.asarray(self.df['mfcc'].to_list()).astype(np.float32)
        
        y = np.asarray(self.df['dig'].to_list()).astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_train, X_test, y_train, y_test


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"]
]:
    try:
        cleaning = DataCleaning(df)
        df = cleaning.clean_data()
        dividing = DataDividing(df)
        X_train, X_test, y_train, y_test = dividing.divide_data()
        logging.info("cleaning completed")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"error cleaning data: {e}")
        raise e