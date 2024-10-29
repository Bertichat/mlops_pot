import logging 
import pandas as pd
import librosa
import numpy as np

from zenml import step

# def amp(x):
#     _ = librosa.stft(x)
#     return _

def amp_db(x):
    k = librosa.amplitude_to_db(abs(librosa.stft(x)))
    print(k.shape)
    return k

def spec_cent(x):
    return librosa.feature.spectral_centroid(y=x, sr=22050)[0]

def mel_spectro(x):
    return librosa.feature.melspectrogram(y=x, sr=22050, n_fft=2048, hop_length=512, n_mels=10)[0]

@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # df["amplitude"] = df["voice_array"].apply(amp)
    df["amplitude_db"] = df["voice_array"].apply(amp_db)
    df["amplitude_db_shape"] = df["amplitude_db"].apply(lambda x: x.shape)
    df["amplitude_db"] = df["amplitude_db"].apply(lambda x: x.flatten())
    df["spectral_centroid"] = df["voice_array"].apply(spec_cent)

    df["mel_spectro"] = df["voice_array"].apply(mel_spectro)

    # df = df.drop(labels=["amplitude",], axis=1)
    return(df)