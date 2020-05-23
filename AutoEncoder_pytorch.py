import torch
import torchvision
from torchvision import transforms, datasets

train_valid = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))
train, valid = torch.utils.data.random_split(train_valid, [55000, 5000])

train_data = torch.utils.data.DataLoader(train, batch_size = 32, shuffle=True)
valid_data = torch.utils.data.DataLoader(valid, batch_size = 5000)
test_data = torch.utils.data.DataLoader(test, batch_size = 10000)

import torch.nn as nn

class MyAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(MyAE, self).__init__()
        self.encoder = nn.Sequential(
        nn.Linear(input_size, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_size), nn.Sigmoid()
        )

    def forward(self, x):
        latent_var = self.encoder(x)
        output = self.decoder(latent_var)
        return output

# Define my AutoEncoder
autoEncoder = MyAE(28*28, 25)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoEncoder.parameters())

# Storages for Losses
Train_Loss = []
Valid_Loss = []

# Training Session
num_epochs = 10
loss_record = 100
for epoch in range(num_epochs):
    for data in train_data:
        inputs, labels = data
        inputs = inputs.view(inputs.size(0), -1) # batch_size x 784

        # Forward Propagate
        outputs = autoEncoder(inputs)
        train_loss = criterion(inputs, outputs)

        # Backward Propagate
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    # Print Last Training Loss of a Single Epoch
    Train_Loss.append(train_loss)
    print("Epoch [{}/{}], Training Loss: {:.4f}".format(epoch+1, num_epochs, train_loss.data))

    for data in valid_data:
        val_inputs, val_labels = data
        val_inputs = val_inputs.view(val_inputs.size(0), -1) # 5000 x 784

        val_outputs = autoEncoder(val_inputs)
        valid_loss = criterion(val_inputs, val_outputs)
    Valid_Loss.append(valid_loss)
    print("epoch [{}/{}], validation loss: {:.4f}".format(epoch + 1, num_epochs, valid_loss.data))
    if valid_loss < loss_record:
        loss_record = valid_loss
        torch.save(autoEncoder.state_dict(), './best_valid.pth')

# Loading a model with the best validation loss during the (num_epochs) epochs
best_AE = MyAE(28*28, 25)
best_AE.load_state_dict(torch.load('./best_valid.pth'))
best_AE.eval()

# Testing Session
for data in test_data:
    test_inputs, test_labels = data
    test_inputs = test_inputs.view(test_inputs.size(0), -1)

    test_outputs = best_AE(test_inputs)
    test_loss = criterion(test_inputs, test_outputs)
print('Test_Loss: {:.4f}'.format(test_loss.data))

import matplotlib.pyplot as plt

x_axis = range(1, num_epochs+1)
plt.plot(x_axis, Train_Loss, 'r-')
plt.plot(x_axis, Valid_Loss, 'b--')
plt.legend(['Training Loss', 'Validation Loss'])
min_loss_idx = Valid_Loss.index(min(Valid_Loss))
plt.plot(min_loss_idx+1, Valid_Loss[min_loss_idx], 'k.')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
