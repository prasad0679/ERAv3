from flask import Flask, render_template, jsonify, request
import json
import os
from threading import Thread

app = Flask(__name__)

# Add these global variables at the top of the file
training_active = True
current_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    from train import train_model  # Import here instead of at the top
    global training_active, current_model
    training_active = True
    data = request.json
    model_id = data['model_id']
    kernel_config = [int(x) for x in data['kernel_config'].split(',')]
    
    # Get configurations for current model
    optimizer_type = data.get('optimizer', 'adam')
    batch_size = int(data.get('batch_size', 512))
    epochs = int(data.get('epochs', 4))
    
    # Save configurations
    try:
        with open('static/model_configs.json', 'w') as f:
            json.dump({
                'model1': {
                    'kernels': data['kernel_config'],
                    'optimizer': data.get('optimizer', 'adam'),
                    'batch_size': data.get('batch_size', 512),
                    'epochs': data.get('epochs', 4)
                },
                'model2': {
                    'kernels': data.get('model2_config', '8,16,32,64'),
                    'optimizer': data.get('model2_optimizer', 'adam'),
                    'batch_size': data.get('model2_batch_size', 512),
                    'epochs': data.get('model2_epochs', 4)
                }
            }, f)
    except Exception as e:
        print(f"Error saving configurations: {e}")
    
    # Start training in a background thread
    current_model = model_id
    thread = Thread(target=train_model, args=(model_id, kernel_config, optimizer_type, batch_size, epochs))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "Training started"})

@app.route('/progress')
def progress():
    try:
        with open('static/progress.json', 'r') as f:
            return jsonify(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({})

@app.route('/predictions')
def predictions():
    try:
        with open('static/predictions.json', 'r') as f:
            return jsonify(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({})

@app.route('/reset_progress', methods=['POST'])
def reset_progress():
    with open('static/progress.json', 'w') as f:
        json.dump({}, f)
    with open('static/predictions.json', 'w') as f:
        json.dump({}, f)
    return jsonify({"status": "Progress reset"})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_active
    training_active = False
    return jsonify({"status": "Stopping training after current epoch"})

@app.route('/training_status')
def training_status():
    return jsonify({
        "active": training_active,
        "current_model": current_model
    })

if __name__ == '__main__':
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Initialize empty files if they don't exist
    if not os.path.exists('static/progress.json'):
        with open('static/progress.json', 'w') as f:
            json.dump({}, f)
    if not os.path.exists('static/predictions.json'):
        with open('static/predictions.json', 'w') as f:
            json.dump({}, f)
    if not os.path.exists('static/model_configs.json'):
        with open('static/model_configs.json', 'w') as f:
            json.dump({
                'model1': '32,64,128,128',
                'model2': '8,16,32,64'
            }, f)
            
    app.run(debug=True) 