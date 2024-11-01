import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import base64
import torchaudio
import torchaudio.transforms as T
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
from utils.visualization import visualize_3d_mesh

class TextDataset(Dataset):
    def __init__(self, text_data):
        # Split the text into lines and convert to lowercase
        self.lines = text_data.strip().split('\n')
        self.lines = [line.lower() for line in self.lines]
        
        # Convert text to tensor
        self.tensors = [torch.tensor([ord(c) for c in line], dtype=torch.long) 
                       for line in self.lines]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.tensors[idx], self.lines[idx]

class ImagePreprocessor:
    def __init__(self):
        # Define standard preprocessing transforms
        self.transforms = {
            'resize': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
            'normalize': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ]),
            'grayscale': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]),
            'color_jitter': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, 
                                    contrast=0.2, 
                                    saturation=0.2, 
                                    hue=0.1),
                transforms.ToTensor(),
            ])
        }
        self.transform_descriptions = {
            'resize': 'Image Resized To Standard Resolution (224x224)',
            'normalize': 'Image Normalized Using ImageNet Standards',
            'grayscale': 'Image Converted To Grayscale Format',
            'color_jitter': 'Image Colors Enhanced For Better Visibility'
        }

    def preprocess_image(self, image):
        # Apply all transforms and return results
        results = {}
        results['labels'] = self.transform_descriptions
        
        # Convert PIL image to RGB if it's in a different mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply each transform and convert result to base64 for display
        for name, transform in self.transforms.items():
            try:
                # Apply transform
                transformed_tensor = transform(image)
                
                # Convert tensor back to PIL Image for display
                if name == 'grayscale':
                    # Handle single channel image
                    transformed_image = transforms.ToPILImage()(transformed_tensor).convert('RGB')
                else:
                    transformed_image = transforms.ToPILImage()(transformed_tensor)
                
                # Convert to base64 for display
                buffered = io.BytesIO()
                transformed_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Store result
                results[name] = {
                    'image_data': f'data:image/png;base64,{img_str}',
                    'tensor_shape': list(transformed_tensor.shape)
                }
            except Exception as e:
                results[name] = f'Error in {name}: {str(e)}'
        
        return results

class AudioPreprocessor:
    def __init__(self):
        self.sample_rate = 16000  # Target sample rate
        self.min_length = 2048    # Minimum length for processing
        
    def save_audio(self, waveform, sample_rate, filename, prefix):
        """Save audio tensor as wav file"""
        try:
            # Ensure waveform is 2D
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            # Convert to float32 if not already
            waveform = waveform.float()
            
            save_path = os.path.join('static/uploads', f'{prefix}_{filename}')
            torchaudio.save(save_path, waveform, sample_rate)
            return f'uploads/{prefix}_{filename}'
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            return None
        
    def pad_audio(self, waveform):
        """Pad audio to minimum length if it's too short"""
        try:
            if waveform.shape[-1] < self.min_length:
                padding_length = self.min_length - waveform.shape[-1]
                padded = torch.nn.functional.pad(waveform, (0, padding_length))
                return padded
            return waveform
        except Exception as e:
            print(f"Error padding audio: {str(e)}")
            return waveform

    def create_waveform_plot(self, waveform, sample_rate, title="Waveform"):
        plt.figure(figsize=(10, 4))
        plt.plot(waveform.t().numpy())
        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Save plot to a base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{plot_data}'

    def preprocess_audio(self, audio_data, filename):
        results = {}
        try:
            print("Starting audio preprocessing...")
            print(f"Input audio data shape: {np.array(audio_data['audio']).shape}")
            print(f"Input sample rate: {audio_data['sample_rate']}")

            # Extract waveform and sample rate from audio data
            waveform = torch.tensor(audio_data['audio'])
            original_sr = audio_data['sample_rate']

            # Store original waveform info
            results['original'] = {
                'waveform_shape': list(waveform.shape),
                'sample_rate': original_sr,
                'duration': waveform.shape[-1] / original_sr,
                'audio_path': f'uploads/{filename}'
            }
            print("Original audio info stored")

            # Ensure audio is mono and 2D
            if len(waveform.shape) > 1 and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            
            print(f"Waveform shape after mono conversion: {waveform.shape}")

            # Initialize resampler if needed
            if original_sr != self.sample_rate:
                print(f"Resampling from {original_sr} to {self.sample_rate}")
                resampler = T.Resample(
                    orig_freq=original_sr,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
                # Save resampled audio
                audio_path = self.save_audio(waveform, self.sample_rate, filename, 'resampled')
                if audio_path:
                    results['resampled'] = {
                        'waveform_shape': list(waveform.shape),
                        'sample_rate': self.sample_rate,
                        'duration': waveform.shape[-1] / self.sample_rate,
                        'audio_path': audio_path
                    }
                    print("Resampled audio saved")

            # Pad audio if too short
            waveform = self.pad_audio(waveform)
            print(f"Waveform shape after padding: {waveform.shape}")
            
            # Save padded audio
            audio_path = self.save_audio(waveform, self.sample_rate, filename, 'padded')
            if audio_path:
                results['padded'] = {
                    'waveform_shape': list(waveform.shape),
                    'sample_rate': self.sample_rate,
                    'duration': waveform.shape[-1] / self.sample_rate,
                    'audio_path': audio_path
                }
                print("Padded audio saved")

            # Apply noise reduction (example preprocessing)
            # Simple noise reduction by applying a threshold
            denoised = waveform.clone()
            threshold = torch.std(denoised) * 0.1
            denoised[torch.abs(denoised) < threshold] = 0
            
            audio_path = self.save_audio(denoised, self.sample_rate, filename, 'denoised')
            if audio_path:
                results['denoised'] = {
                    'waveform_shape': list(denoised.shape),
                    'sample_rate': self.sample_rate,
                    'duration': denoised.shape[-1] / self.sample_rate,
                    'audio_path': audio_path
                }
                print("Denoised audio saved")

            # Add visualization for original waveform
            results['original']['plot'] = self.create_waveform_plot(
                waveform, 
                original_sr, 
                "Original Waveform"
            )

            # After resampling
            if 'resampled' in results:
                results['resampled']['plot'] = self.create_waveform_plot(
                    waveform, 
                    self.sample_rate, 
                    "Resampled Waveform"
                )

            # After padding
            if 'padded' in results:
                results['padded']['plot'] = self.create_waveform_plot(
                    waveform, 
                    self.sample_rate, 
                    "Padded Waveform"
                )

            # After denoising
            if 'denoised' in results:
                results['denoised']['plot'] = self.create_waveform_plot(
                    denoised, 
                    self.sample_rate, 
                    "Denoised Waveform"
                )

            # Add labels for each version
            results['labels'] = {
                'original': 'Original Audio Signal',
                'resampled': f'Audio Resampled To {self.sample_rate}Hz For Better Quality',
                'padded': 'Audio Padded To Standard Length',
                'denoised': 'Audio With Reduced Background Noise'
            }

        except Exception as e:
            print(f"Error in audio preprocessing: {str(e)}")
            results['error'] = str(e)

        return results

# Add this class for 3D preprocessing
class MeshPreprocessor:
    def __init__(self):
        self.transform_descriptions = {
            'normalized': 'Mesh Scaled To Unit Sphere',
            'centered': 'Mesh Centered At Origin'
        }

    def normalize_mesh(self, vertices):
        """Scale mesh to unit sphere"""
        # Calculate center and scale
        center = torch.mean(vertices, dim=0)
        vertices = vertices - center
        scale = torch.max(torch.norm(vertices, dim=1))
        vertices = vertices / scale
        return vertices

    def center_mesh(self, vertices):
        """Center mesh at origin"""
        center = torch.mean(vertices, dim=0)
        return vertices - center

    def create_mesh_visualization(self, vertices, faces, title):
        """Create visualization with adjusted size"""
        fig = plt.figure(figsize=(8, 8))  # Adjusted figure size
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the mesh surface
        surf = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                             triangles=faces,
                             cmap='viridis',
                             alpha=0.4)
        
        # Plot vertices as red points
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  color='red', 
                  s=50,
                  alpha=1.0,
                  label='Vertices')
        
        # Plot edges
        for face in faces:
            points = vertices[face]
            for i in range(3):
                p1 = points[i]
                p2 = points[(i + 1) % 3]
                ax.plot([p1[0], p2[0]], 
                       [p1[1], p2[1]], 
                       [p1[2], p2[2]], 
                       color='#0066FF',
                       linewidth=4.0,
                       alpha=1.0)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend(['Surface', 'Vertices', 'Edges'])
        ax.view_init(elev=30, azim=45)
        
        # Save plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{plot_data}'

    def preprocess_mesh(self, vertices, faces):
        results = {}
        results['labels'] = self.transform_descriptions

        try:
            # Normalize mesh
            normalized_vertices = self.normalize_mesh(vertices)
            normalized_plot = self.create_mesh_visualization(
                normalized_vertices.numpy(), 
                faces.numpy(), 
                "Normalized Mesh"
            )
            results['normalized'] = {
                'vertices': normalized_vertices.numpy().tolist(),  # Convert to list
                'faces': faces.numpy().tolist(),  # Convert to list
                'plot': normalized_plot
            }

            # Center mesh
            centered_vertices = self.center_mesh(vertices)
            centered_plot = self.create_mesh_visualization(
                centered_vertices.numpy(), 
                faces.numpy(), 
                "Centered Mesh"
            )
            results['centered'] = {
                'vertices': centered_vertices.numpy().tolist(),  # Convert to list
                'faces': faces.numpy().tolist(),  # Convert to list
                'plot': centered_plot
            }

        except Exception as e:
            print(f"Error in mesh preprocessing: {str(e)}")
            results['error'] = str(e)

        return results

def preprocess_data(data, file_type, filename=None):
    """
    Preprocessing operations
    Args:
        data: The loaded data
        file_type: Type of the file ('text', 'image', 'audio', '3d')
        filename: Original filename (needed for audio processing)
    Returns:
        Preprocessed data
    """
    if file_type == 'text':
        # Create dataset from text
        dataset = TextDataset(data)
        
        # Process and return results
        processed_result = {
            'original_text': data,
            'processed_lines': dataset.lines,
            'labels': {
                'original': 'Original Text Content',
                'processed': 'Text Converted To Lowercase For Standardization'
            }
        }
        return processed_result
    
    elif file_type == 'image':
        # Create image preprocessor and process image
        preprocessor = ImagePreprocessor()
        if isinstance(data, dict) and 'image_data' in data:
            return preprocessor.preprocess_image(data['image_data'])
        else:
            return preprocessor.preprocess_image(data)
    
    elif file_type == 'audio':
        # Create audio preprocessor and process audio
        preprocessor = AudioPreprocessor()
        return preprocessor.preprocess_audio(data, filename)
    
    elif file_type == '3d':
        preprocessor = MeshPreprocessor()
        vertices = data['vertices']
        faces = data['faces']
        return preprocessor.preprocess_mesh(vertices, faces)