# Import necessary libraries
import torch  # Core library for PyTorch
import torch.nn as nn  # For building neural networks
import torch.optim as optim  # For optimization algorithms
import torchvision.transforms as transforms  # For image transformations
import torchvision.datasets as datasets  # For standard datasets
from torchvision.utils import save_image  # For saving images
import numpy as np  # For numerical operations
import os  # For interacting with the operating system

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder part of U-Net: Downsampling layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # Convolution layer
            nn.ReLU(),  # Activation function
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Convolution layer
            nn.ReLU()  # Activation function
        )
        # Middle part of U-Net: Bottleneck layers
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Convolution layer
            nn.ReLU(),  # Activation function
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Convolution layer
            nn.ReLU()  # Activation function
        )
        # Decoder part of U-Net: Upsampling layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Transpose convolution layer
            nn.ReLU(),  # Activation function
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # Transpose convolution layer
            nn.Tanh()  # Activation function, scales output between -1 and 1
        )

    def forward(self, x):
        x = self.encoder(x)  # Apply encoder
        x = self.middle(x)  # Apply middle part
        x = self.decoder(x)  # Apply decoder
        return x

# Instantiate the U-Net model and move it to the appropriate device
model = UNet().to(device)

# Define the diffusion process
class Diffusion:
    def __init__(self, num_steps=1000):
        self.num_steps = num_steps  # Number of diffusion steps
        self.beta = np.linspace(0.0001, 0.02, num_steps)  # Linear schedule for beta
        self.alpha = 1.0 - self.beta  # Calculate alpha from beta
        self.alpha_bar = np.cumprod(self.alpha)  # Cumulative product of alphas

    # Forward diffusion process: add noise to image
    def q_sample(self, x_start, t, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise  # Generate noise if not provided
        return torch.sqrt(self.alpha_bar[t]) * x_start + torch.sqrt(1 - self.alpha_bar[t]) * noise

    # Reverse diffusion process: denoise image
    def p_sample(self, x_t, t, model):
        x_t = x_t.to(device)  # Move input to the correct device
        t = t.to(device)  # Move time step to the correct device
        noise = torch.randn_like(x_t)  # Generate noise
        predicted_noise = model(x_t, t)  # Predict noise using the model
        return torch.sqrt(1 / self.alpha[t]) * (x_t - ((1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_bar[t])) * predicted_noise) + noise

# Instantiate the diffusion process
diffusion = Diffusion()

# Set hyperparameters
batch_size = 64  # Batch size for training
learning_rate = 1e-4  # Learning rate for optimizer
num_epochs = 10  # Number of training epochs

# Load and transform the dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize images to 28x28
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Download and load the MNIST dataset
dataset = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss

# Training loop
for epoch in range(num_epochs):
    for idx, (x, _) in enumerate(dataloader):
        x = x.to(device)  # Move input to the correct device
        t = torch.randint(0, diffusion.num_steps, (batch_size,), device=device).long()  # Random time steps

        noise = torch.randn_like(x)  # Generate noise
        x_t = diffusion.q_sample(x, t, noise)  # Apply forward diffusion
        predicted_noise = model(x_t, t)  # Predict noise with the model
        
        loss = criterion(predicted_noise, noise)  # Calculate loss
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        # Print loss every 100 batches
        if idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

# Function to generate and save images
def sample_images(model, diffusion, num_samples=64):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        x = torch.randn(num_samples, 1, 28, 28).to(device)  # Generate random noise
        for t in reversed(range(diffusion.num_steps)):  # Reverse diffusion process
            t_batch = torch.tensor([t] * num_samples, device=device).long()  # Time steps batch
            x = diffusion.p_sample(x, t_batch, model)  # Denoise
        save_image(x, os.path.join('samples', f'sample.png'), normalize=True)  # Save generated images

# Create samples directory if it does not exist
if not os.path.exists('samples'):
    os.makedirs('samples')

# Generate and save images from the trained model
sample_images(model, diffusion)