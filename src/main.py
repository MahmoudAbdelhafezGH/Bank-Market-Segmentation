from dataPreprocessing import preprocess
from evaluation import evaluate
from featureExtraction import features
from models import model
from training import train
from utils import utils

DATA_PATH = 'data/raw/data.csv'

# Step 1: preprocess the data
df = preprocess.load_data(DATA_PATH)
df = preprocess.clean_data(df)
df = preprocess.encode_data(df)

# Step 2: Standarize and reduce dimensions
df_scaled = features.scale_features(df)
df_reduced = features.reduce_dimensions(df_scaled)

# Step 3: Create Classification Model
classifier = model.NeuralNetwork()

# Step 4: Train
train_loader, test_loader = utils.prepare_train_test_loader(df_reduced)
train.train(classifier, train_loader)

# Step 5: Test Model
evaluate.evaluate(classifier, test_loader)