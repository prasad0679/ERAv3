import torch
from torch.utils.data import Dataset
import nltk
from nltk.corpus import wordnet
import random
from textblob import TextBlob
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import base64
import torchaudio
import torchaudio.transforms as T
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
from utils.visualization import visualize_3d_mesh

# Text augmentation setup
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

class TextAugmenter:
    def __init__(self):
        pass

    def get_synonyms(self, word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        return list(set(synonyms))

    def synonym_replacement(self, text, n=1):
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if len(self.get_synonyms(word)) > 0]))
        
        n = min(n, len(random_word_list))
        if n == 0:
            return text
            
        for _ in range(n):
            random_word = random.choice(random_word_list)
            random_synonym = random.choice(self.get_synonyms(random_word))
            random_idx = random.randint(0, len(new_words)-1)
            new_words[random_idx] = random_synonym
            
        return ' '.join(new_words)

    def random_insertion(self, text, n=1):
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            add_word = random.choice(words)
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, add_word)
            
        return ' '.join(new_words)

    def augment_text(self, data):
        lines = data.strip().split('\n')
        
        augmented_results = {
            'original_text': data,
            'augmentations': {
                'synonym_replacement': [],
                'random_insertion': [],
            },
            'labels': {
                'synonym_replacement': 'Text With Words Replaced By Their Synonyms',
                'random_insertion': 'Text With Additional Words For Context Enhancement'
            }
        }
        
        for line in lines:
            syn_replaced = self.synonym_replacement(line, n=2)
            random_inserted = self.random_insertion(line, n=2)
            
            augmented_results['augmentations']['synonym_replacement'].append(syn_replaced)
            augmented_results['augmentations']['random_insertion'].append(random_inserted)
        
        return augmented_results

class ImageAugmenter:
    def __init__(self):
        # Define different augmentation transforms
        self.transforms = {
            'horizontal_flip': transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
            ]),
            'vertical_flip': transforms.Compose([
                transforms.RandomVerticalFlip(p=1.0),
                transforms.ToTensor(),
            ]),
            'rotation': transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
            ]),
            'color_jitter': transforms.Compose([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
                transforms.ToTensor(),
            ]),
            'random_crop': transforms.Compose([
                transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                transforms.ToTensor(),
            ]),
            'gaussian_blur': transforms.Compose([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
            ])
        }
        self.transform_descriptions = {
            'horizontal_flip': 'Image Flipped Left To Right',
            'vertical_flip': 'Image Flipped Top To Bottom',
            'rotation': 'Image Rotated By 30 Degrees',
            'color_jitter': 'Image Colors Randomly Adjusted',
            'random_crop': 'Image Randomly Cropped And Resized',
            'gaussian_blur': 'Image Smoothed With Gaussian Blur'
        }

    def augment_image(self, image):
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
                transformed_image = transforms.ToPILImage()(transformed_tensor)
                
                # Convert to base64 for display
                buffered = io.BytesIO()
                transformed_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Store result
                results[name] = {
                    'image_data': f'data:image/png;base64,{img_str}',
                    'description': self.get_transform_description(name)
                }
            except Exception as e:
                results[name] = f'Error in {name}: {str(e)}'
        
        return results

    def get_transform_description(self, transform_name):
        descriptions = {
            'horizontal_flip': 'Flips the image horizontally (left to right)',
            'vertical_flip': 'Flips the image vertically (upside down)',
            'rotation': 'Rotates the image randomly up to 30 degrees',
            'color_jitter': 'Randomly adjusts brightness, contrast, saturation, and hue',
            'random_crop': 'Randomly crops and resizes a portion of the image',
            'gaussian_blur': 'Applies Gaussian blur with random sigma'
        }
        return descriptions.get(transform_name, 'No description available')

class AudioAugmenter:
    def __init__(self):
        self.sample_rate = 16000
        self.min_length = 2048

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
        if waveform.shape[-1] < self.min_length:
            padding_length = self.min_length - waveform.shape[-1]
            padded = torch.nn.functional.pad(waveform, (0, padding_length))
            return padded
        return waveform

    def time_stretch(self, waveform, rate=1.2):
        """Time stretching"""
        try:
            effects = [
                ["tempo", f"{rate}"],
            ]
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, effects)
            return augmented
        except Exception as e:
            print(f"Error in time stretch: {str(e)}")
            return waveform

    def pitch_shift(self, waveform, n_steps=2):
        """Pitch shifting"""
        try:
            effects = [
                ["pitch", f"{n_steps * 100}"],
                ["rate", f"{self.sample_rate}"]
            ]
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, self.sample_rate, effects)
            return augmented
        except Exception as e:
            print(f"Error in pitch shift: {str(e)}")
            return waveform

    def add_noise(self, waveform, noise_level=0.005):
        """Add Gaussian noise"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise

    def reverse(self, waveform):
        """Reverse the audio"""
        return torch.flip(waveform, [-1])

    def augment_audio(self, audio_data, filename):
        results = {}
        try:
            # Extract waveform and sample rate
            waveform = torch.tensor(audio_data['audio'])
            original_sr = audio_data['sample_rate']

            # Ensure audio is mono and 2D
            if len(waveform.shape) > 1 and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            elif len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)

            # Resample if needed
            if original_sr != self.sample_rate:
                resampler = T.Resample(orig_freq=original_sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # Pad if too short
            waveform = self.pad_audio(waveform)

            # Store original info
            results['original'] = {
                'waveform_shape': list(waveform.shape),
                'sample_rate': self.sample_rate,
                'duration': waveform.shape[-1] / self.sample_rate,
                'audio_path': f'uploads/{filename}'
            }

            # Time stretch
            stretched = self.time_stretch(waveform)
            audio_path = self.save_audio(stretched, self.sample_rate, filename, 'time_stretched')
            if audio_path:
                results['time_stretched'] = {
                    'waveform_shape': list(stretched.shape),
                    'sample_rate': self.sample_rate,
                    'duration': stretched.shape[-1] / self.sample_rate,
                    'audio_path': audio_path
                }

            # Pitch shift
            pitched = self.pitch_shift(waveform)
            audio_path = self.save_audio(pitched, self.sample_rate, filename, 'pitch_shifted')
            if audio_path:
                results['pitch_shifted'] = {
                    'waveform_shape': list(pitched.shape),
                    'sample_rate': self.sample_rate,
                    'duration': pitched.shape[-1] / self.sample_rate,
                    'audio_path': audio_path
                }

            # Add noise
            noisy = self.add_noise(waveform)
            audio_path = self.save_audio(noisy, self.sample_rate, filename, 'noisy')
            if audio_path:
                results['noisy'] = {
                    'waveform_shape': list(noisy.shape),
                    'sample_rate': self.sample_rate,
                    'duration': noisy.shape[-1] / self.sample_rate,
                    'audio_path': audio_path
                }

            # Reverse
            reversed_audio = self.reverse(waveform)
            audio_path = self.save_audio(reversed_audio, self.sample_rate, filename, 'reversed')
            if audio_path:
                results['reversed'] = {
                    'waveform_shape': list(reversed_audio.shape),
                    'sample_rate': self.sample_rate,
                    'duration': reversed_audio.shape[-1] / self.sample_rate,
                    'audio_path': audio_path
                }

            # Add descriptions
            results['labels'] = {
                'original': 'Original Audio Signal',
                'time_stretched': 'Audio Slowed Down By 20%',
                'pitch_shifted': 'Audio Pitch Raised By 2 Steps',
                'noisy': 'Audio With Added Random Noise',
                'reversed': 'Audio Played In Reverse'
            }

            # Add visualization for original waveform
            results['original']['plot'] = create_waveform_plot(
                waveform, 
                self.sample_rate, 
                "Original Waveform"
            )

            # Add plots for each augmentation
            if 'time_stretched' in results:
                results['time_stretched']['plot'] = create_waveform_plot(
                    stretched, 
                    self.sample_rate, 
                    "Time Stretched Waveform"
                )

            if 'pitch_shifted' in results:
                results['pitch_shifted']['plot'] = create_waveform_plot(
                    pitched, 
                    self.sample_rate, 
                    "Pitch Shifted Waveform"
                )

            if 'noisy' in results:
                results['noisy']['plot'] = create_waveform_plot(
                    noisy, 
                    self.sample_rate, 
                    "Noisy Waveform"
                )

            if 'reversed' in results:
                results['reversed']['plot'] = create_waveform_plot(
                    reversed_audio, 
                    self.sample_rate, 
                    "Reversed Waveform"
                )

        except Exception as e:
            print(f"Error in audio augmentation: {str(e)}")
            results['error'] = str(e)

        return results

# Add this class for 3D augmentation
class MeshAugmenter:
    def __init__(self):
        self.transform_descriptions = {
            'rotated': 'Mesh Rotated By 45 Degrees Around Y-Axis',
            'scaled': 'Mesh Scaled By Factor Of 1.5',
            'noisy': 'Mesh With Added Random Vertex Noise'
        }

    def rotate_mesh(self, vertices, angle_degrees=45):
        """Rotate mesh around Y-axis"""
        angle_rad = torch.tensor(angle_degrees * np.pi / 180.0)
        rot_matrix = torch.tensor([
            [torch.cos(angle_rad), 0, torch.sin(angle_rad)],
            [0, 1, 0],
            [-torch.sin(angle_rad), 0, torch.cos(angle_rad)]
        ], dtype=vertices.dtype)
        
        return torch.matmul(vertices, rot_matrix)

    def scale_mesh(self, vertices, scale_factor=1.5):
        """Scale mesh uniformly"""
        return vertices * scale_factor

    def add_noise(self, vertices, noise_factor=0.05):
        """Add random noise to vertices"""
        noise = torch.randn_like(vertices) * noise_factor
        return vertices + noise

    def augment_mesh(self, vertices, faces):
        results = {}
        results['labels'] = self.transform_descriptions

        try:
            # Rotate mesh
            rotated_vertices = self.rotate_mesh(vertices)
            rotated_plot = visualize_3d_mesh(
                rotated_vertices.numpy(), 
                faces.numpy(), 
                "Rotated Mesh"
            )
            results['rotated'] = {
                'vertices': rotated_vertices.numpy().tolist(),
                'faces': faces.numpy().tolist(),
                'plot': rotated_plot
            }

            # Scale mesh
            scaled_vertices = self.scale_mesh(vertices)
            scaled_plot = visualize_3d_mesh(
                scaled_vertices.numpy(), 
                faces.numpy(), 
                "Scaled Mesh"
            )
            results['scaled'] = {
                'vertices': scaled_vertices.numpy().tolist(),
                'faces': faces.numpy().tolist(),
                'plot': scaled_plot
            }

            # Add noise
            noisy_vertices = self.add_noise(vertices)
            noisy_plot = visualize_3d_mesh(
                noisy_vertices.numpy(), 
                faces.numpy(), 
                "Noisy Mesh"
            )
            results['noisy'] = {
                'vertices': noisy_vertices.numpy().tolist(),
                'faces': faces.numpy().tolist(),
                'plot': noisy_plot
            }

        except Exception as e:
            print(f"Error in mesh augmentation: {str(e)}")
            results['error'] = str(e)

        return results

# Update the augment_data function to include 3D augmentation
def augment_data(data, file_type, filename=None):
    if file_type == 'text':
        augmenter = TextAugmenter()
        return augmenter.augment_text(data)
    
    elif file_type == 'image':
        augmenter = ImageAugmenter()
        if isinstance(data, dict) and 'image_data' in data:
            return augmenter.augment_image(data['image_data'])
        else:
            return augmenter.augment_image(data)
    
    elif file_type == 'audio':
        augmenter = AudioAugmenter()
        return augmenter.augment_audio(data, filename)
    
    elif file_type == '3d':
        augmenter = MeshAugmenter()
        vertices = data['vertices']
        faces = data['faces']
        return augmenter.augment_mesh(vertices, faces)

def create_waveform_plot(waveform, sample_rate, title="Waveform"):
    plt.figure(figsize=(6, 2))  # Smaller figure size
    plt.plot(waveform.t().numpy())
    plt.title(title, fontsize=10)  # Smaller font size
    plt.xlabel("Sample", fontsize=8)
    plt.ylabel("Amplitude", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to fit
    
    # Save plot to a base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{plot_data}'