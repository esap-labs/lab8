import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.models import LinearNet

if __name__ == '__main__':

    # Set the batch size
    batch_size = 4096

    # Check if CUDA is available (Note: It is not currently benificial to use GPU acceleration of the Raspberry Pi)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device: {}".format(device))

    # Load the MNIST dataset
    transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.FashionMNIST('/home/pi/ee347/lab8/data', train=True, download=True, transform=transform)
    testset = datasets.FashionMNIST('/home/pi/ee347/lab8/data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Start timer
    t = time.time_ns()

    # Create the model and optimizer
    model = LinearNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    model.train()
    train_loss = 0

    # Batch Loop
    for i, (images, labels) in enumerate(tqdm(train_loader, leave=False)):

        # Move the data to the device (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute the loss
        loss = F.nll_loss(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()

        # Accumulate the loss
        train_loss = train_loss + loss.item()

    # Test the model
    model.eval()
    correct = 0
    total = 0

    # Batch Loop
    for images, labels in tqdm(test_loader, leave=False):

        # Move the data to the device (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Get the predicted class from the maximum value in the output-list of class scores
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        # Accumulate the number of correct classifications
        correct += (predicted == labels).sum().item()

    # Print the epoch statistics
    print("Training Loss: {:.4f}, Test Accuracy: {:.2f}%, Time Taken: {:.2f}s".format(train_loss / len(train_loader), 100 * correct / total, (time.time_ns() - t) / 1e9))