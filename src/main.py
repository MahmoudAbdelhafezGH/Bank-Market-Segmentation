from dataPreprocessing import preprocess
from evaluation import evaluate
from featureExtraction import features
from models import model
from training import train
from utils import utils

DATA_PATH = 'data/raw/data.csv'

def main():

    # Step 1: preprocess the data
    df = preprocess.load_data(DATA_PATH)
    df_cleaned = preprocess.clean_data(df)
    df_encoded = preprocess.encode_data(df_cleaned)

    # Step 2: Standarize and reduce dimensions
    df_scaled = features.scale_features(df_encoded)
    df_reduced = features.reduce_dimensions(df_scaled)

    # Step 3: Create Classification Model
    classifier = model.NeuralNetwork()

    # Step 4: Train
    train_loader, test_loader = utils.split_data(df_reduced)
    train.train(classifier, train_loader)

    # Step 5: Test Model
    evaluate.evaluate(classifier, test_loader)

if __name__ == "__main__":
    main()