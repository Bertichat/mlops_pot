from zenml import pipeline
from steps.import_data import extract_data
from steps.cleaning import clean_data
from steps.model_training import train_model
from steps.evaluation import evaluate_model

@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    df = extract_data(data_path)
    df = clean_data(df)
    # train_model(df)
    # evaluate_model(df)