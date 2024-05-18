from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn
import torch.optim as optim
import torch.nn.init as init
from sklearn.metrics import confusion_matrix
bank_marketing = fetch_ucirepo(id=222) 
  
df = bank_marketing.data.original
df.dropna(subset = ['y'], inplace = True)
catgeorical_columns = []
numeric_columns = []

df.drop(['contact', 'poutcome', 'duration'], axis = 1, inplace=True)
df.dropna(inplace=True)

for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        if column not in ['previous', 'pdays']:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR
            upper_array = np.where(df[column] >= upper)[0]
            lower_array = np.where(df[column] <= lower)[0]
            upper_array = [idx for idx in upper_array if idx in df.index]
            lower_array = [idx for idx in lower_array if idx in df.index]
            df = df[(df[column] >= lower) & (df[column] <= upper)]

df_encoded_target = df.copy()

def one_hot_encode(df, column_name):
    if column_name in df.columns:
        df_encoded = pd.get_dummies(df[column_name], prefix=column_name, dtype=int)
        df.drop(column_name, axis=1, inplace=True)
        df = pd.concat([df, df_encoded], axis=1)
        print("One-hot encoding applied to column ", column_name, " successfully.")
        return df
    else:
        print(f"Column '{column_name}' not found in the DataFrame.")


df_encoded = df.replace(['yes', 'no'], [1, 0])
df_encoded.replace(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 
                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)
df_encoded = one_hot_encode(df_encoded, 'job')
df_encoded = one_hot_encode(df_encoded, 'education')
df_encoded = one_hot_encode(df_encoded, 'marital')
df_max_scaled = df_encoded.copy()  
# apply normalization techniques 
for column in df_max_scaled.columns:
    try:
        df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max()
    except:
        raise Exception(f'Error by trying to scale column {column}. Values: {df_encoded[column].unique()}' )

print(df_encoded['y'].isna().sum(), df_max_scaled['y'].isna().sum(), df['y'].isna().sum())

dim_red = PCA()
x_pca = dim_red.fit_transform(df_max_scaled.drop('y', axis=1))
training_features = ['pca0', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9']
df_pca = pd.DataFrame(x_pca, columns=dim_red.get_feature_names_out())
df_pca_working_set = df_pca[training_features]
df_pca_working_set = df_pca_working_set.reset_index(drop=True)
df_reset_index = df_max_scaled.reset_index(drop=True)
df_pca_working_set['y'] = df_reset_index['y']

learning_rate = .0001
epochs_num = 100
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)

    def forward(self, inputs):
        return self.layers(inputs)

import pdb
pdb.set_trace()

non_numeric_columns = df.select_dtypes(exclude=['float64', 'float32', 'int64', 'int32']).columns.tolist()
# Drop columns with non-numeric data types
df_numeric = df.drop(columns=non_numeric_columns)

NN = NeuralNetwork()
cost = nn.BCELoss()
optimizer = optim.Adam(NN.parameters(), lr = learning_rate)

x_tensor = torch.tensor(df_pca_working_set.drop(columns=['y']).tail(10000).values, dtype=torch.float32)
y_tensor = torch.tensor(df_pca_working_set['y'].tail(10000).values, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
train_loader = torch.utils.data.DataLoader(dataset, batch_size = 8)
for epoch in range(1000):
    total_training_loss = 0
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data
        targets = targets.unsqueeze(1)
        optimizer.zero_grad()
        preds = NN.forward(inputs)
        loss = cost(preds, targets)
        total_training_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch}. Total Training Loss: ', total_training_loss)

x_test_df = df_pca_working_set.drop(columns='y').iloc[20000: 25000, ]
y_test_df = df_pca_working_set['y'].iloc[10000: 15000, ]
x_test_tensor = torch.tensor(x_test_df.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_df.values, dtype=torch.float32)
test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 8)

y_pred_list = []
y_true = np.empty((0, ))
y_pred = np.empty((0, ))

for inputs, labels in test_loader:
        output = NN(inputs) # Feed Network
        output = (output > 0.5).float()
        output = output.detach().cpu().numpy()
        y_pred = np.append(y_pred, output)
        
        labels = labels.data.cpu().numpy()
        y_true =  np.append(y_true, labels) # Save Truth


cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
print(cm)