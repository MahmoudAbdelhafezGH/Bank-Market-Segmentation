import torch

def split_data(df):

    x_tensor = torch.tensor(df.drop(columns=['y']).tail(7000).values, dtype=torch.float32)

    y_tensor = torch.tensor(df['y'].tail(7000).values, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 64)

    x_test_tensor = torch.tensor(df.drop(columns=['y']).head(1000).values, dtype=torch.float32)

    y_test_tensor = torch.tensor(df['y'].head(1000).values, dtype=torch.float32)

    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 8)

    return train_loader, test_loader