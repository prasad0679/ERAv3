from flask import Flask, render_template, request, jsonify, url_for
import os
from utils.preprocessing import preprocess_data
from utils.augmentation import augment_data
import torch
from PIL import Image
import soundfile as sf
import trimesh
import base64
from io import BytesIO
import torchaudio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from utils.visualization import visualize_3d_mesh

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {
    'text': {'txt', 'csv', 'json'},
    'image': {'png', 'jpg', 'jpeg'},
    'audio': {'wav', 'mp3'},
    '3d': {'obj', 'stl', 'ply', 'off'}
}

def allowed_file(filename, file_type):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

@app.route('/')
def index():
    print("Index route accessed")  # Debug print
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering index: {str(e)}")  # Debug print
        return str(e), 500  # Return error to browser

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    file_type = request.form.get('file_type')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not allowed_file(file.filename, file_type):
        return jsonify({'error': 'Invalid file type'})
    
    # Save file
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Load file based on type
    data = load_file(filepath, file_type)
    
    # For images, create a data URL
    if file_type == 'image':
        data = {
            'image_url': url_for('static', filename=f'uploads/{filename}'),
            'image_data': data
        }
    
    return render_template('display.html', 
                         filename=filename,
                         file_type=file_type,
                         data=data)

@app.route('/process', methods=['POST'])
def process_data():
    try:
        filename = request.form.get('filename')
        file_type = request.form.get('file_type')
        operation = request.form.get('operation')
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        data = load_file(filepath, file_type)
        
        if operation == 'preprocess':
            result = preprocess_data(data, file_type, filename)
        else:  # augment
            result = augment_data(data, file_type, filename)
        
        # Ensure we're returning valid JSON
        if result is None:
            return jsonify({'error': 'Processing failed'})
            
        return jsonify({'result': result})
        
    except Exception as e:
        print(f"Error in process_data: {str(e)}")  # Debug print
        return jsonify({'error': str(e)})

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Handle cleanup request from home button"""
    try:
        filename = request.form.get('filename')
        if filename:
            # Get all files in uploads directory
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                # Check if file is a preprocessed or augmented version
                if file.startswith(('resampled_', 'padded_', 'denoised_', 
                                  'time_stretched_', 'pitch_shifted_', 
                                  'noisy_', 'reversed_')):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                    try:
                        os.remove(file_path)
                        print(f"Removed file: {file}")
                    except Exception as e:
                        print(f"Error removing file {file}: {str(e)}")
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def load_file(filepath, file_type):
    if file_type == 'text':
        with open(filepath, 'r') as f:
            return f.read()
    
    elif file_type == 'image':
        image = Image.open(filepath)
        return image
    
    elif file_type == 'audio':
        try:
            print(f"Loading audio file: {filepath}")
            waveform, sr = torchaudio.load(filepath)
            
            # Create visualization for original audio
            from utils.preprocessing import AudioPreprocessor
            preprocessor = AudioPreprocessor()
            plot = preprocessor.create_waveform_plot(waveform, sr, "Original Audio Signal")
            
            return {
                'audio': waveform.numpy(), 
                'sample_rate': sr,
                'plot': plot
            }
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            try:
                audio, sr = sf.read(filepath)
                return {'audio': audio, 'sample_rate': sr}
            except Exception as e:
                print(f"Error loading audio with soundfile: {str(e)}")
                return None
    
    elif file_type == '3d':
        try:
            # Load the mesh using trimesh
            mesh = trimesh.load(filepath)
            vertices = torch.tensor(mesh.vertices, dtype=torch.float)
            faces = torch.tensor(mesh.faces, dtype=torch.long)
            
            # Create visualization
            plot = visualize_3d_mesh(mesh.vertices, mesh.faces)
            
            return {
                'vertices': vertices,
                'faces': faces,
                'plot': plot,
                'label': "3D Mesh Visualization"
            }
        except Exception as e:
            print(f"Error loading 3D file: {str(e)}")
            return None

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000) 