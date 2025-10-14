""" Simple example models for testing
"""
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def train_and_save(classifier:torch.nn.Module, train_dataloader:DataLoader,
                   model_pth:Path=Path('./model/'), device='cuda', epochs:int=10):
    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(epochs):  # Train for 10 epochs
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = classifier(images)  # Forward pass
            loss = loss_fn(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

        logger.info("Epoch: %d, loss is %f", epoch, loss.item())

    # Save the trained model
    torch.save(classifier.state_dict(), model_pth)
    logger.info("Model state saved in: %s", model_pth)

def onnx_export(classifier:torch.nn.Module, model_pth:Path=Path('./model/onnx_model.onnx'),
                input_shape=(1, 1, 28, 28), device='cuda'):
    """ Export the model to ONNX format.
    """
    classifier.eval()  # Set the model to evaluation mode
    dummy_input = torch.randn(input_shape, requires_grad=False).to(device)  # Create a dummy input tensor
    torch.onnx.export(classifier,
                      dummy_input,   # type:ignore
                      model_pth,
                      export_params=True,  # Store the trained parameter weights inside the model file
                      opset_version=10,  # ONNX version to export to
                      do_constant_folding=True,  # Optimize constant folding
                      input_names=['input'],  # Name of the input layer
                      output_names=['output'],  # Name of the output layer
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})  # Dynamic batch size

# Define the image classifier model
class ImageClassifierCNN(torch.nn.Module):
    """ Simple MNIST digits classifier based on CNN.
    """
    def __init__(self):
        super(ImageClassifierCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.ReLU()
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class ImageClassifierLinear(torch.nn.Module):
    """ Simple MNIST digits classifier based on fully-conected linear layers.
    """
    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(784,128)
        self.hidden2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64,10)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        return self.output(x)
