import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
# Load and preprocess the dataset
#transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat single channel grayscale image to get three channel RGB image
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)


# Create the Vision Transformer model
def create_vit_model():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=10, img_size=32)
    return model


vit_model = create_vit_model()

# Specify the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_model.parameters(), lr=1e-3)

# Train the model
for epoch in range(20):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = vit_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Train Epoch: {epoch} \tLoss: {loss.item()}')

# Test the model
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = vit_model(data)
        test_loss += criterion(output, target).item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print(f'Test loss: {test_loss}, Test accuracy: {correct / len(test_loader.dataset)}')
