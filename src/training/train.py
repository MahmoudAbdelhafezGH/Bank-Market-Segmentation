import torch
from torch import nn
from torch import optim

EPOCHS = 3000
LEARNING_RATE = .001

def train(model, train_loader):
    
    learning_rate = LEARNING_RATE
    
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    cost = nn.MSELoss()

    for epoch in range(EPOCHS):

        total_training_loss = 0

        for i, data in enumerate(train_loader, 0):

            inputs, targets = data

            targets = targets.unsqueeze(1)

            optimizer.zero_grad()

            preds = model.forward(inputs)

            weight = torch.where(targets == 1, torch.tensor([1.0]), torch.tensor([0.02]))
            
            loss = cost(preds, targets) * weight
            
            loss = loss.mean()
            
            total_training_loss += loss.item()

            loss.backward()

            optimizer.step()

        if (epoch%500) == 0 and epoch != 0:
            
            learning_rate = learning_rate / 2
            
            for param in optimizer.param_groups:
            
                param['lr'] = learning_rate

        print(f'Epoch {epoch}. Learning Rate: {learning_rate} Total Training Loss: ', total_training_loss)

    return model

