import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms 
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__() #inherit from nn.Module

        # each conv layer is followed by a relu activation and a max pooling layer
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x): # x is a vector 
        x = F.relu(self.conv1(x)) # relu activation == adds non-linearity, ie. sharpens
        x = self.pool(x) # max pooling
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # restructures tensor into one dimension
        return x
    
device = "cuda" if torch.cuda.is_available() else "cpu"

input_size = 784 # 28x28 images flattened
num_classes = 10 # digits 0-9
learning_rate = 0.001
batch_size = 64
num_epochs = 10

train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# will pull random sample from trainnig dataset

labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()





model = CNN(in_channels=1, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        # move data and targets to device
        data = data.to(device)
        targets = targets.to(device)

        # forward: compute model output and loss
        scores = model(data)
        loss = criterion(scores, targets)

        # backward : compute gradient of the loss with respect to model parameters
        optimizer.zero_grad()
        loss.backward()

        # optimize: update model parameters
        optimizer.step()

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    
    num_correct = 0
    num_samples = 0
    model.eval() # set model to evaluation mode

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            #forward
            scores = model(x)
            _, predictions = scores.max(1) # get the index of the max log-probability
            num_correct += (predictions == y).sum() # count correct predictions
            num_samples += predictions.size(0) # count total samples

        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}")

    model.train() # set model back to training mode

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
