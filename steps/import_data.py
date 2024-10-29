import logging 
import pandas as pd
import multiprocessing as mp

from zenml import step

from sklearn.datasets import load_iris

# class ImportData:
#     def __init__(self, data_path: str):
#         self.data_path = data_path

#     def get_data(self):
#         logging.info(f"importing data from {self.data_path}")
#         return pd.read_csv(self.data_path)
    
# @step
# def import_data(data_path: str) -> pd.DataFrame:
#     try :
#         importing_data = ImportData(data_path)
#         df = importing_data.get_data()
#         return df
#     except Exception as e:
#         logging.error(f"Error while importing data: {e}")
#         raise e


import numpy as np
import glob
import os
import sys
import scipy.io.wavfile as wavf
import scipy.signal
import json
import librosa
import multiprocessing
import argparse

NB_FOLDER = 2
NB_PROCESS = 1

@step
def extract_data(src, n_processes=NB_PROCESS):
    folders = []

    for folder in os.listdir(src):
        # only process folders
        if not os.path.isdir(os.path.join(src, folder)):
            continue
        elif int(folder) <= NB_FOLDER:
            folders.append(folder)
    print(folders)
    df = pd.DataFrame(columns=["dig", "vp", "rep", "sr", "voice_array"])
    pool=multiprocessing.Pool(processes=n_processes)
    df_list=pool.map(_preprocess_data, 
               [os.path.join(src, folder) for folder in sorted(folders)])
    
    df = pd.concat(df_list)
    df.to_csv('df.csv', index=True)
    return df
    

def _preprocess_data(src):

    df_temp = pd.DataFrame(columns=["dig", "vp", "rep", "sr", "voice_array"])
    print("processing {}".format(src))
    # loop over recordings
    for filepath in sorted(glob.glob(os.path.join(src, "*.wav"))):
        dig, vp, rep = filepath.rstrip(".wav").split("/")[-1].split("_")
        print(dig)
        print(vp)
        print(rep)
        voice_array, sr = librosa.load(filepath)
        df_temp.loc[len(df_temp)] = [dig, vp, rep, sr, voice_array]

    df_temp.to_csv("df_temp.csv", index=True)
    return df_temp