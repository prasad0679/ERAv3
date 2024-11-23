import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.model import MNISTNet
from src.utils import evaluate_model
from src.train import train_model

def test_model_parameters():
    model = MNISTNet()
    assert model.count_parameters() < 25000, "Model has too many parameters"

def test_input_shape():
    model = MNISTNet()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Model output shape is incorrect"

def test_model_accuracy():
    model = MNISTNet()
    # Train for a few batches
    device = torch.device('cpu')
    model = train_model(device, save_suffix='test')
    
    accuracy = evaluate_model(model, device)
    assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below threshold" 