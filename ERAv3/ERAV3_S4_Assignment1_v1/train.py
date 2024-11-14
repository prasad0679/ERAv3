import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import os
from model import MNIST_CNN
import random
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Check CUDA availability and properties
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
    logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
logger.info("Loading MNIST dataset...")
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory=True)
logger.info(f"Dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# Initialize model, loss function, and optimizer
model = MNIST_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def train():
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    
    logger.info("Starting training...")
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Create progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/10', ncols=100)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
            
            if batch_idx % 10 == 9:
                avg_loss = running_loss / 10
                train_losses.append(avg_loss)
                running_loss = 0.0
                
                # Calculate current training accuracy
                train_acc = 100 * correct_train / total_train
                train_accuracies.append(train_acc)
                
                # Save training progress
                progress = {
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'test_accuracies': test_accuracies
                }
                with open('static/progress.json', 'w') as f:
                    json.dump(progress, f)
                
                # Update progress bar description
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'train_acc': f'{train_acc:.2f}%'
                })
        
        # Calculate final training accuracy for the epoch
        train_acc = 100 * correct_train / total_train
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing', leave=False):
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_acc = 100 * correct / total
        test_accuracies.append(test_acc)
        
        # Log epoch results
        logger.info(
            f'Epoch {epoch+1} - '
            f'Train Accuracy: {train_acc:.2f}% - '
            f'Test Accuracy: {test_acc:.2f}% - '
            f'Average Loss: {avg_loss:.4f}'
        )
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_mnist_cnn.pth')
            logger.info(f'New best model saved with accuracy: {best_accuracy:.2f}%')

    # Save final model
    torch.save(model.state_dict(), 'final_mnist_cnn.pth')
    logger.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    
    # Generate and save predictions for 10 random test images
    generate_random_predictions()

def generate_random_predictions():
    logger.info("Generating predictions for random test images...")
    model.eval()
    random_indices = random.sample(range(len(test_dataset)), 10)
    predictions = []
    
    with torch.no_grad():
        for idx in random_indices:
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            pred = output.argmax(dim=1, keepdim=True).item()
            
            img_np = image.cpu().squeeze().numpy()
            
            predictions.append({
                'image': img_np.tolist(),
                'prediction': pred,
                'actual': label
            })
    
    with open('static/predictions.json', 'w') as f:
        json.dump(predictions, f)
    logger.info("Predictions generated and saved")

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    train() 