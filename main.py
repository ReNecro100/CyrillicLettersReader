import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

print("PyTorch version:", torch.__version__)
epochs = 20
learning_rate = 0.0005
size_of_a_batch = 16

augmentation_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    torchvision.transforms.RandomRotation(degrees = 30),
    transforms.ToTensor()
    ])

train_dataset = torchvision.datasets.MNIST(root="/", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.MNIST(root="/", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=size_of_a_batch, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=size_of_a_batch, shuffle=False)


print('\n\n'+str(len(train_dataset)))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(nn.Linear(784,1000), nn.ReLU(), nn.Linear(1000,3000), nn.ReLU(), nn.Linear(3000,10)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []

print(device)
for num_epochs in range(epochs):
  print(f"№{num_epochs+1}: GO")
  for i, (images, labels) in tqdm(enumerate(train_dataloader)):
    images, labels = images.to(device), labels.to(device)
    images = images.view(size_of_a_batch, 784).to(device)
    output = model(images)
    loss = criterion(output, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_dataloader:
      images, labels = images.to(device), labels.to(device)
      images = images.view(size_of_a_batch, 784).to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    losses.append((correct/total)*100)

print(losses)