import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

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
import matplotlib.pyplot as plt
import base64
from io import BytesIO

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
    for epoch in range(4):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Create progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/4', ncols=100)
        
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
        
        # Save progress after each epoch's test accuracy
        progress = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
        with open('static/progress.json', 'w') as f:
            json.dump(progress, f)
        
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
            
            # Get the original image and properly scale it
            original_image = test_dataset.data[idx].cpu().numpy()
            # Scale to 0-255 range and invert colors
            original_image = 255 - (original_image * (255 / original_image.max()))
            original_image = original_image.astype(np.uint8)
            
            predictions.append({
                'image': original_image.tolist(),
                'prediction': int(pred),
                'actual': int(label)
            })
            
            # Debug log
            logger.info(f"Image shape: {original_image.shape}, Range: [{original_image.min()}, {original_image.max()}]")
    
    with open('static/predictions.json', 'w') as f:
        json.dump(predictions, f)
    logger.info("Predictions generated and saved")

def get_training_active():
    # Import here to avoid circular import
    from server import training_active
    return training_active

def train_model(model_id, kernel_config, optimizer_type='adam', batch_size=512, num_epochs=4):
    logger.info(f"Starting training for {model_id} with config: kernels={kernel_config}, optimizer={optimizer_type}, batch_size={batch_size}, epochs={num_epochs}")
    
    # Initialize model
    model = MNIST_CNN(kernel_config).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer based on type
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Create data loaders with specified batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory=True)
    
    # Load or initialize progress data
    try:
        with open('static/progress.json', 'r') as f:
            progress_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        progress_data = {}
    
    # Initialize progress for this model
    progress_data[model_id] = {
        "training_loss": [],
        "train_accuracy": [],
        "test_accuracy": []
    }
    
    # Training loop
    for epoch in range(int(num_epochs)):  # Convert num_epochs to int and use it
        if not get_training_active():
            logger.info(f"Training stopped for {model_id} after epoch {epoch}")
            break
            
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f'{model_id} Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            if not get_training_active():
                break
                
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
                train_acc = 100 * correct_train / total_train
                
                # Update progress data
                progress_data[model_id]["training_loss"].append(avg_loss)
                progress_data[model_id]["train_accuracy"].append(train_acc)
                
                # Save progress
                with open('static/progress.json', 'w') as f:
                    json.dump(progress_data, f)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'acc': f'{train_acc:.2f}%'
                })
                
                running_loss = 0.0
        
        # Test accuracy after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_acc = 100 * correct / total
        progress_data[model_id]["test_accuracy"].append(test_acc)
        
        # Save progress
        with open('static/progress.json', 'w') as f:
            json.dump(progress_data, f)
        
        logger.info(f'{model_id} Epoch {epoch+1}/{num_epochs} - Test Accuracy: {test_acc:.2f}%')
    
    # Generate predictions after training is complete
    generate_predictions_for_model(model, model_id)
    
    # If this was model1 and training wasn't stopped, start model2
    if model_id == 'model1' and get_training_active():
        logger.info("Model 1 training complete, starting Model 2...")
        try:
            with open('static/model_configs.json', 'r') as f:
                configs = json.load(f)
                model2_config = [int(x) for x in configs['model2']['kernels'].split(',')]
                model2_optimizer = configs['model2']['optimizer']
                model2_batch_size = int(configs['model2']['batch_size'])
                model2_epochs = int(configs['model2']['epochs'])
                train_model('model2', model2_config, model2_optimizer, model2_batch_size, model2_epochs)
        except Exception as e:
            logger.error(f"Error starting Model 2 training: {e}")
            logger.error(f"Model 2 configs: {configs['model2']}")  # Add debug logging

def generate_predictions_for_model(model, model_id):
    logger.info(f"Generating predictions for {model_id}")
    model.eval()
    random_indices = random.sample(range(len(test_dataset)), 10)
    
    predictions_data = {}
    try:
        with open('static/predictions.json', 'r') as f:
            predictions_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions_data = {}
    
    predictions_data[model_id] = []
    
    with torch.no_grad():
        for idx in random_indices:
            image, label = test_dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()
            
            # Get the original image
            img_array = test_dataset.data[idx].numpy()
            
            # Create a clear figure with white background
            plt.figure(figsize=(2, 2), dpi=100)
            plt.imshow(img_array, cmap='gray')
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # Save to base64 with white background
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                       facecolor='white', edgecolor='none', dpi=100)
            plt.close()
            buf.seek(0)
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            predictions_data[model_id].append({
                'image': img_base64,
                'prediction': int(pred),
                'actual': int(label)
            })
    
    # Save predictions atomically
    with open('static/predictions.json', 'w') as f:
        json.dump(predictions_data, f)
    
    logger.info(f"Predictions generated and saved for {model_id}")

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    train() 