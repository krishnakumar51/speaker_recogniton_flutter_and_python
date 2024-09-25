import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import soundfile as sf
from speechbrain.pretrained import EncoderClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import librosa
import random
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Constants
AUTHORIZED_USER_FOLDER = "rea"
UNAUTHORIZED_USER_FOLDER = "nt"
NOISE_FOLDER = "noise"
MODEL_PATH = "finetuned_ecapa_model.pth"

class FineTunedECAPATDNN(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()

        print(pretrained_model.mods)
        self.feature_extractor = pretrained_model.mods['compute_features']
        self.norm = pretrained_model.mods['mean_var_norm']
        self.encoder = pretrained_model.mods['embedding_model']
        
        # We will set the classifier after knowing the shape of the encoder output
        self.classifier = None
    
    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = self.norm(feats, torch.ones(feats.shape[0], device=feats.device))
        embeddings = self.encoder(feats)
        
        # Print the shape of embeddings
        if self.classifier is None:
            print(f"Encoder output shape: {embeddings.shape}")  # Debugging line to see the shape
            # Set classifier input dimension based on embeddings' shape
            embedding_dim = embeddings.shape[-1]  # Extracting the last dimension (192 in this case)
            self.classifier = nn.Linear(embedding_dim, 2)  # Binary classification: 2 output classes
        
        # Flatten the encoder's output from (batch_size, 1, embedding_dim) to (batch_size, embedding_dim)
        embeddings = embeddings.squeeze(1)  # Removes the extra dimension (1 in the middle)

        output = self.classifier(embeddings)
        return output



def load_audio(file_path, max_length=16000):  # 1 second at 16kHz
    waveform, sr = librosa.load(file_path, sr=16000)
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    else:
        waveform = np.pad(waveform, (0, max_length - len(waveform)))
    return torch.tensor(waveform).float()

def augment_data(audio_path, output_folder):
    """Augment audio data by applying pitch shifting, time stretching, and adding noise.
       Save each augmented audio as a separate file.
    """
    # Base name for the augmented files
    base_filename = os.path.basename(audio_path).split('.')[0]

    # Load audio
    signal, sr = sf.read(audio_path)
    if len(signal.shape) > 1:
        signal = signal[:, 0]  # Take only the first channel if it's stereo

    # Pitch Shift
    pitch_shift = random.randint(-3, 3)  # Shift pitch between -3 and +3 semitones
    pitch_shifted_signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=pitch_shift)
    pitch_shifted_path = os.path.join(output_folder, f"{base_filename}_pitch_shift_{pitch_shift}.wav")
    sf.write(pitch_shifted_path, pitch_shifted_signal, sr)
    print(f"Pitch-shifted audio saved at {pitch_shifted_path}")

    # Time Stretch
    stretch_factor = random.uniform(0.8, 1.2)  # Stretch time by 80% to 120%
    time_stretched_signal = librosa.effects.time_stretch(signal, rate=stretch_factor)
    time_stretched_path = os.path.join(output_folder, f"{base_filename}_time_stretch_{stretch_factor:.2f}.wav")
    sf.write(time_stretched_path, time_stretched_signal, sr)
    print(f"Time-stretched audio saved at {time_stretched_path}")

    # Add Noise
    noise_files = [os.path.join(NOISE_FOLDER, f) for f in os.listdir(NOISE_FOLDER) if f.endswith('.wav')]
    noise_signal, _ = sf.read(random.choice(noise_files))  # Randomly pick a noise file

    # If noise is stereo, convert to mono by averaging the channels
    if len(noise_signal.shape) > 1:
        noise_signal = noise_signal.mean(axis=1)  # Convert to mono by averaging
    
    # Pad or truncate the noise signal to match the length of the original signal
    noise_signal = np.pad(noise_signal, (0, len(signal) - len(noise_signal))) if len(noise_signal) < len(signal) else noise_signal[:len(signal)]
    
    # Add noise
    signal_with_noise = signal + 0.005 * noise_signal  # Add noise with a small factor
    noise_augmented_path = os.path.join(output_folder, f"{base_filename}_with_noise.wav")
    sf.write(noise_augmented_path, signal_with_noise, sr)
    print(f"Noise-augmented audio saved at {noise_augmented_path}")



def load_audio(file_path, max_length=16000):  # e.g., 1 second at 16kHz
    waveform, sr = librosa.load(file_path, sr=16000)
    if len(waveform) > max_length:
        waveform = waveform[:max_length]
    else:
        waveform = np.pad(waveform, (0, max_length - len(waveform)), mode='constant')
    return torch.tensor(waveform).float()

# Modify the preprocessing to ensure all audio samples are of the same length
def preprocess_audio(audio_path, target_length=16000):
    """Preprocess audio by reducing noise, extracting non-silent segments, and padding/truncating to a fixed length."""
    signal, sr = sf.read(audio_path)
    
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=signal, sr=sr)
    
    # Convert to AudioSegment for VAD
    audio_segment = AudioSegment(reduced_noise.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    
    # Apply VAD (Voice Activity Detection)
    chunks = split_on_silence(audio_segment, min_silence_len=500, silence_thresh=-40)
    
    # Concatenate non-silent chunks
    voice_only = sum(chunks)
    
    # Convert back to numpy array
    voice_array = np.array(voice_only.get_array_of_samples())
    
    # Pad or truncate to ensure fixed length
    if len(voice_array) > target_length:
        voice_array = voice_array[:target_length]
    else:
        voice_array = np.pad(voice_array, (0, target_length - len(voice_array)), 'constant')
    
    return voice_array, sr

# Modify the training data preparation to handle padding/truncating of audio samples
def train_speaker_model(authorized_user_folder, unauthorized_user_folder):
    print("Loading pre-trained ecapa-TDNN model...")
    pretrained = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                                savedir="pretrained_models/spkrec-ecapa-voxceleb")
    
    model = FineTunedECAPATDNN(pretrained)
    
    # Prepare data
    print("Preparing data for fine-tuning...")
    authorized_files = [os.path.join(authorized_user_folder, f) for f in os.listdir(authorized_user_folder) if f.endswith('.wav')]
    unauthorized_files = [os.path.join(unauthorized_user_folder, f) for f in os.listdir(unauthorized_user_folder) if f.endswith('.wav')]
    
    # Augment authorized and unauthorized data
    for file in authorized_files + unauthorized_files:
        augment_data(file, authorized_user_folder if file in authorized_files else unauthorized_user_folder)
    
    # Preprocess and load audio files
    target_length = 16000  # Fixed length for all audio samples
    X_authorized = [torch.tensor(preprocess_audio(f, target_length=target_length)[0]).float() for f in authorized_files]
    X_unauthorized = [torch.tensor(preprocess_audio(f, target_length=target_length)[0]).float() for f in unauthorized_files]
    
    y_authorized = torch.ones(len(X_authorized))  # Authorized speaker label
    y_unauthorized = torch.zeros(len(X_unauthorized))  # Unauthorized speaker label
    
    X = X_authorized + X_unauthorized
    y = torch.cat([y_authorized, y_unauthorized])
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 50

    dataset = TensorDataset(torch.stack(X_train), y_train)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print("Starting fine-tuning...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_train_predictions = 0  # To count correct predictions for training accuracy
        total_train_samples = 0  # To keep track of total samples in the training set
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # Pass the entire batch
            
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Compute training accuracy for this batch
            predicted_labels = torch.argmax(outputs, dim=1)
            correct_train_predictions += (predicted_labels == targets).sum().item()
            total_train_samples += targets.size(0)  # Add the number of samples in this batch
        
        # Training accuracy for the epoch
        train_accuracy = correct_train_predictions / total_train_samples
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = [model(x.unsqueeze(0)) for x in X_val]
            val_loss = sum(criterion(output, y.unsqueeze(0).long()) for output, y in zip(val_outputs, y_val)).item()
            val_accuracy = sum((torch.argmax(output, dim=1) == y.long()).float() for output, y in zip(val_outputs, y_val)).item() / len(y_val)
        
        # Print both training and validation metrics
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(X_train):.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    
    print("Fine-tuning completed. Saving model...")
    torch.save(model.state_dict(), MODEL_PATH)
    return model




# def recognize_speaker_with_wake_word(audio_path, model):
#     model.eval()
#     with torch.no_grad():
#         waveform = torch.tensor(preprocess_audio(audio_path)[0]).float()
#         output = model(waveform.unsqueeze(0)).squeeze(1) 
#         probability = torch.softmax(output, dim=1)[0][1].item()
    
#     threshold = 0.8  # You may need to adjust this
#     is_authorized = probability >= threshold
    
#     if is_authorized:
#         return "Speaker identified", probability
#     else:
#         return "Speaker not detected", probability


def recognize_speaker_with_wake_word(audio_path, model):
    model.eval()
    with torch.no_grad():
        waveform = torch.tensor(preprocess_audio(audio_path)[0]).float()
        
        # Pass the waveform through the model
        output = model(waveform.unsqueeze(0))
        
        # Ensure the output has the right shape and apply softmax
        output = output.squeeze(1)  # Flatten if there's an extra dimension
        probabilities = torch.softmax(output, dim=1)  # Apply softmax to the output
        
        # Get the probability of the authorized class (index 1)
        probability = probabilities[0][1].item()
    
    # Adjust the threshold as needed
    threshold = 0.5  # Try lowering the threshold temporarily
    is_authorized = probability >= threshold
    
    if is_authorized:
        return "Speaker identified", probability
    else:
        return "Speaker not detected", probability

# Load or train the model
if os.path.exists(MODEL_PATH):
    print("Loading fine-tuned model...")
    pretrained = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                                savedir="pretrained_models/spkrec-ecapa-voxceleb")
    model = FineTunedECAPATDNN(pretrained)
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)
else:
    print("Fine-tuned model not found. Training new model...")
    model = train_speaker_model(AUTHORIZED_USER_FOLDER, UNAUTHORIZED_USER_FOLDER)


# Test the model
test_audio_path = "rek.wav"
is_authorized, probability = recognize_speaker_with_wake_word(test_audio_path, model)
if is_authorized:
    print(f"Speaker identified as authorized with probability: {probability:.4f}")
else:
    print(f"Speaker identified as unauthorized. Probability of being authorized: {probability:.4f}")