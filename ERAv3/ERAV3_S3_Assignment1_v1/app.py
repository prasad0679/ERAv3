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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {
    'text': {'txt', 'csv', 'json'},
    'image': {'png', 'jpg', 'jpeg'},
    'audio': {'wav', 'mp3'},
    '3d': {'obj', 'stl', 'ply'}
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
    filename = request.form.get('filename')
    file_type = request.form.get('file_type')
    operation = request.form.get('operation')
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = load_file(filepath, file_type)
    
    if operation == 'preprocess':
        result = preprocess_data(data, file_type, filename)
    else:  # augment
        result = augment_data(data, file_type, filename)
    
    return jsonify({'result': result})

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
            return {'audio': waveform.numpy(), 'sample_rate': sr}
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            # Fallback to soundfile if torchaudio fails
            try:
                audio, sr = sf.read(filepath)
                return {'audio': audio, 'sample_rate': sr}
            except Exception as e:
                print(f"Error loading audio with soundfile: {str(e)}")
                return None
    
    elif file_type == '3d':
        mesh = trimesh.load(filepath)
        return {
            'vertices': mesh.vertices.tolist(),
            'faces': mesh.faces.tolist()
        }

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000) 