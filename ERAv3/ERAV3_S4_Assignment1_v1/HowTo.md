# MNIST CNN Training with Real-time Visualization

This project implements a 4-layer CNN for MNIST digit classification with real-time training visualization through a web interface.

## Requirements 
''' 
pip install torch torchvision flask matplotlib numpy

'''
## Project Structure

mnist_cnn/
├── HowTo.md
├── train.py
├── model.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
└── server.py 

## How to Run

1. Start the Flask server:

''' 
python server.py
'''
2. In a separate terminal, start the training:

''' 
python train.py
'''
3. Open your browser and go to:

''' 
http://127.0.0.1:5000
'''
You will see the real-time training progress, loss curves, and after training completes, predictions on random test images.
