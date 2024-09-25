import os
import torch
import numpy as np
import soundfile as sf
from speechbrain.pretrained import SpeakerRecognition, Tacotron2
from sklearn.metrics.pairwise import cosine_similarity
# import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import split_on_silence
from speechbrain.inference.ASR import EncoderDecoderASR
import time
import librosa

# Define constants
# NOISE_FOLDER = "noise"
# WAKE_WORD = "hi siri"  # Set the wake word
SPEAKER="speaker"
EMBEDDINGS_FILE_PATH = "speaker/authorized_embedding_avg.npy"

# Load models
spk_rec_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                                savedir="pretrained_models/spkrec-ecapa-voxceleb")
# asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", 
#                                              savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")

# def augment_data(audio_path, output_folder):
#     """Augment audio data by adding noise and saving it to the output folder."""
#     # Load audio
#     signal, sr = sf.read(audio_path)

#     # Add noise from NOISE_FOLDER
#     for noise_file in os.listdir(NOISE_FOLDER):
#         noise_path = os.path.join(NOISE_FOLDER, noise_file)
#         noise_signal, _ = sf.read(noise_path)
        
#         # Ensure noise is the same length as the original audio
#         if len(noise_signal) > len(signal):
#             noise_signal = noise_signal[:len(signal)]
#         elif len(noise_signal) < len(signal):
#             noise_signal = np.pad(noise_signal, (0, max(0, len(signal) - len(noise_signal))), mode='constant')

#         # Control noise addition
#         noise_factor = np.random.uniform(0.01, 0.1)  # Random noise level
#         augmented_signal = signal + noise_factor * noise_signal
        
#         # Make sure to normalize the augmented signal
#         augmented_signal = np.clip(augmented_signal, -1.0, 1.0)  # Assuming signal is in the range [-1.0, 1.0]

#         # Save augmented audio
#         augmented_path = os.path.join(output_folder, f"augmented_{os.path.basename(audio_path)}")
#         sf.write(augmented_path, augmented_signal, sr)
#         print(f"Augmented audio saved at {augmented_path}")


# def preprocess_audio(audio_path):
#     """Preprocess audio by reducing noise and extracting non-silent segments."""
#     signal, sr = sf.read(audio_path)
    
#     # Apply noise reduction
#     reduced_noise = nr.reduce_noise(y=signal, sr=sr)
    
#     # Convert to AudioSegment for VAD
#     audio_segment = AudioSegment(reduced_noise.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    
#     # Apply VAD
#     chunks = split_on_silence(audio_segment, min_silence_len=500, silence_thresh=-40)
    
#     # Concatenate non-silent chunks
#     voice_only = sum(chunks)
    
#     # Convert back to numpy array
#     voice_array = np.array(voice_only.get_array_of_samples())
    
#     return voice_array, sr


# def augment_data(audio_path, output_folder):
#     """Augment audio data by applying pitch shifting and time stretching, then saving it to the output folder."""
#     print(f"Starting augmentation for: {audio_path}")
    
#     # Load audio
#     signal, sr = sf.read(audio_path)
#     print(f"Original signal shape: {signal.shape}, Sample rate: {sr}")

#     # Ensure the original signal is 1D (mono)
#     if len(signal.shape) > 1:
#         signal = signal[:, 0]  # Take only the first channel if it's stereo

#     # List to hold augmented signals
#     augmented_signals = []

#     # Pitch Shift
#     pitch_shift = random.randint(-3, 3)  # Shift pitch between -3 and +3 semitones
#     pitch_shifted_signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=pitch_shift)
#     augmented_signals.append(pitch_shifted_signal)
#     print(f"Applied pitch shift of {pitch_shift} semitones.")

#     # Time Stretch
#     stretch_factor = random.uniform(0.8, 1.2)  # Stretch time by 80% to 120%
    
#     try:
#         # Time stretch operation with librosa
#         time_stretched_signal = librosa.effects.time_stretch(signal, stretch_factor)
#         augmented_signals.append(time_stretched_signal)
#         print(f"Applied time stretch with factor: {stretch_factor:.2f}")
#     except TypeError as e:
#         print(f"Error applying time stretch: {e}")
#         return  # Early return if time stretching fails

#     # Combine all augmented signals by averaging
#     if augmented_signals:
#         combined_signal = np.mean(augmented_signals, axis=0)
#     else:
#         combined_signal = signal  # Fallback to original if no augmentations

#     # Normalize the combined signal
#     combined_signal = np.clip(combined_signal, -1.0, 1.0)

#     # Save augmented audio
#     augmented_path = os.path.join(output_folder, f"augmented_{os.path.basename(audio_path)}")
#     sf.write(augmented_path, combined_signal, sr)
#     print(f"Augmented audio saved at {augmented_path}")



# Ensure you call this function appropriately in your app






def get_embedding(audio_path):
    """Get speaker embedding from the audio."""
    signal, fs = librosa.load(audio_path, sr=None)
    signal = torch.tensor(np.expand_dims(signal, axis=0))
    embeddings = spk_rec_model.encode_batch(signal)
    return embeddings.squeeze().cpu().detach().numpy()
    # signal, sr = preprocess_audio(audio_path)
    

# def preprocess_audio(audio_path):
#     """Preprocess audio by reducing noise and extracting non-silent segments."""
#     signal, sr = sf.read(audio_path)
    
#     # Apply noise reduction
#     reduced_noise = nr.reduce_noise(y=signal, sr=sr)
    
#     # Convert to AudioSegment for VAD
#     audio_segment = AudioSegment(reduced_noise.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    
#     # Apply VAD
#     chunks = split_on_silence(audio_segment, min_silence_len=300, silence_thresh=-40)  # Adjust parameters as needed
    
#     # Concatenate non-silent chunks
#     if chunks:
#         voice_only = sum(chunks)
#     else:
#         print("No voice detected.")
#         return np.array([]), sr  # Return empty array if no voice detected

#     # Convert back to numpy array
#     voice_array = np.array(voice_only.get_array_of_samples())
    
#     return voice_array, sr

# def register_authorized_speaker(authorized_folder):
#     """Register an authorized speaker by averaging their embeddings."""
#     authorized_embeddings = []
#     for file_name in os.listdir(authorized_folder):
#         file_path = os.path.join(authorized_folder, file_name)
#         embedding = get_embedding(file_path)
#         authorized_embeddings.append(embedding)
#     return np.mean(authorized_embeddings, axis=0)

def is_authorized_speaker(new_audio_path, authorized_embedding_avg, threshold=0.75):
    """Check if the new audio matches the authorized speaker embedding."""
    new_embedding = get_embedding(new_audio_path)
    similarity = cosine_similarity([new_embedding], [authorized_embedding_avg])[0][0]
    print(f"Cosine Similarity: {similarity}")
    return similarity >= threshold, similarity

def recognize_speaker_with_wake_word(audio_path, authorized_embedding_avg):
    """Recognize the speaker with a wake word detection."""
    try:
        signal, sr = sf.read(audio_path)
        print(f"Audio file read successfully. Shape: {signal.shape}, Sample rate: {sr}")

        # # Use the ASR model for transcription
        # transcription = asr_model.transcribe_file(audio_path)
        # print(f"Transcription: {transcription}")

        # if WAKE_WORD.lower() in transcription.lower():
        #     print(f"Wake word '{WAKE_WORD}' detected.")
        is_authorized, similarity = is_authorized_speaker(audio_path, authorized_embedding_avg)
            
        if is_authorized:
            print("Speaker identified!")
            return "Speaker identified", similarity
        # else:
        #     print("Wake word detected, but speaker not identified.")
        #     return "Wake word detected, but speaker not identified", similarity
        # else:
        #     print(f"Wake word '{WAKE_WORD}' not detected.")
        #     return "Wake word not detected", None
        else:
            print(f"Speaker not detected.")
            return "Speaker not detected", similarity
    except Exception as e:
        print(f"Error in recognize_speaker_with_wake_word: {str(e)}")
        raise  # Re-raise the exception to be caught by the outer try-except blocks


def register_authorized_speaker(authorized_folder):
    """Register an authorized speaker by averaging their embeddings."""
    authorized_embeddings = []
    total_files = len([f for f in os.listdir(authorized_folder) if f.endswith('.wav')])
    processed_files = 0
    
    print(f"Found {total_files} audio files for processing.")
    
    for file_name in os.listdir(authorized_folder):
        if file_name.endswith('.wav'):
            start_time = time.time()
            file_path = os.path.join(authorized_folder, file_name)
            embedding = get_embedding(file_path)
            authorized_embeddings.append(embedding)
            processed_files += 1
            processing_time = time.time() - start_time
            print(f"Processed {file_name} ({processed_files}/{total_files}) in {processing_time:.2f} seconds")
    
    if not authorized_embeddings:
        print("No embeddings were generated. Check if there are valid audio files in the folder.")
        return None
    
    return np.mean(authorized_embeddings, axis=0)

def train_speaker_model(authorized_user_folder):
    """Train the speaker model using registered speakers' audio samples."""
    print(f"Starting training with files from {authorized_user_folder}")
    start_time = time.time()
    authorized_embedding_avg = register_authorized_speaker(authorized_user_folder)
    total_time = time.time() - start_time
    
    if authorized_embedding_avg is not None:
        print(f"Training completed in {total_time:.2f} seconds")
        save_embeddings(authorized_embedding_avg, EMBEDDINGS_FILE_PATH) 
        return authorized_embedding_avg
    else:
        print("Training failed: No valid embeddings generated")
        return None


def save_embeddings(embeddings, file_path):
    """Save the embeddings to a file."""
    np.save(file_path, embeddings)
    print(f"Embeddings saved to {file_path}")

def load_embeddings(file_path):
    """Load the embeddings from a file."""
    if os.path.exists(file_path):
        embeddings = np.load(file_path)
        print(f"Embeddings loaded from {file_path}")
        return embeddings
    else:
        print(f"No embeddings file found at {file_path}")
        return None