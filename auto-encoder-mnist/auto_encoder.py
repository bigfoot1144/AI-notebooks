
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
import tqdm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
import pygame
import numpy as np 

NUM_SLIDERS = 10
# Train the model
epochs = 4
batch_size = 16

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)

# Split the data into training and testing sets
X_train, X_test = mnist.data[:60000], mnist.data[60000:]
y_train, y_test = mnist.target[:60000], mnist.target[60000:]

# Normalize the data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train = torch.tensor(X_train.values)
X_test = torch.tensor(X_test.values)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_SLIDERS),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(NUM_SLIDERS, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encoder_only(self, x):
        x = self.encoder(x)
        return x
    
    def decoder_only(self,x):
        x = self.decoder(x)
        return x

model = Autoencoder().to("cuda")
print(model)
X_train = X_train.to("cuda")

optim = torch.optim.Adam(model.parameters(),lr=3e-4)
loss_fn = nn.MSELoss()
num_batches = len(X_train) // batch_size

with tqdm.tqdm(total=epochs * num_batches, desc="Training") as pbar:
    for epoch in range(epochs):
        model.train()
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            inputs = X_train[start_idx:end_idx]
            
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)
            loss.backward()
            optim.step()
            
            pbar.update(1)
            pbar.set_postfix({"Epoch": epoch+1, "Batch": batch_idx+1, "Loss": f"{loss.item():.4f}"})

# Constants
WIDTH, HEIGHT = 800, 600
SLIDER_WIDTH, SLIDER_HEIGHT = 600, 20
IMAGE_SIZE = 280
SLIDER_MIN = -100
SLIDER_MAX = 100

model = model.to("cpu")

# Initialize input image
input_image = np.random.rand(1, 784)
input_image = torch.tensor(input_image).float()

# Get initial slider values from encoder
initial_slider_values = model.encoder_only(input_image).squeeze(0).cpu().detach().numpy()

# Initialize Pygame
pygame.init()

# Set font
font = pygame.font.Font(None, 24)

# Slider properties
sliders = []
for i in range(NUM_SLIDERS):
    slider = {"x": 100, "y": 50 + i * 30, "value": initial_slider_values[i], "min": SLIDER_MIN, "max": SLIDER_MAX}
    sliders.append(slider)

# Function to draw sliders
def draw_sliders():
    for i, slider in enumerate(sliders):
        pygame.draw.rect(screen, (200, 200, 200), (slider["x"], slider["y"], SLIDER_WIDTH, SLIDER_HEIGHT))
        slider_pos = int(SLIDER_WIDTH * (slider["value"] - slider["min"]) / (slider["max"] - slider["min"]))
        pygame.draw.rect(screen, (100, 100, 100), (slider["x"] + slider_pos, slider["y"], 5, SLIDER_HEIGHT))
        text = font.render(f"Value {i}: {slider['value']:.2f}", True, (0, 0, 0))
        screen.blit(text, (slider["x"], slider["y"] - 20))

# Function to update sliders
def update_sliders(pos, clicked):
    for slider in sliders:
        if slider["x"] < pos[0] < slider["x"] + SLIDER_WIDTH and slider["y"] < pos[1] < slider["y"] + SLIDER_HEIGHT and clicked:
            slider["value"] = max(SLIDER_MIN, min(SLIDER_MAX, (pos[0] - slider["x"]) / SLIDER_WIDTH * (SLIDER_MAX - SLIDER_MIN) + SLIDER_MIN))
            return True
    return False

# Create window
screen = pygame.display.set_mode((WIDTH, HEIGHT + (NUM_SLIDERS * 30)))

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get mouse position and button state
    mouse_pos = pygame.mouse.get_pos()
    mouse_clicked = pygame.mouse.get_pressed()[0]  # Left mouse button

    # Update sliders
    update_sliders(mouse_pos, mouse_clicked)

    # Draw background
    screen.fill((255, 255, 255))

    # Draw sliders
    draw_sliders()

    # Get new output image
    new_slider_values = np.array([slider["value"] for slider in sliders])
    new_slider_values = torch.tensor(new_slider_values).float()
    new_output = model.decoder_only(new_slider_values.to("cpu"))
    new_output_image = new_output.cpu().detach().numpy().reshape((28, 28))

    # Draw output image
    output_image_surface = pygame.Surface((28, 28))
    for y in range(28):
        for x in range(28):
            val = int(new_output_image[y, x] * 255)
            output_image_surface.set_at((x, y), (val, val, val))
    output_image_surface = pygame.transform.scale(output_image_surface, (IMAGE_SIZE, IMAGE_SIZE))
    screen.blit(output_image_surface, (WIDTH // 2 - IMAGE_SIZE // 2, HEIGHT // 2 - IMAGE_SIZE // 2 + (NUM_SLIDERS * 30)))

    # Update display
    pygame.display.flip()

pygame.quit()