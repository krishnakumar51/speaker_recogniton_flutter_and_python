import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
from speechbrain.pretrained import SpeakerRecognition
from sklearn.model_selection import train_test_split

# Constants
AUTHORIZED_FOLDER = "authenticated_user"
UNAUTHORIZED_FOLDER = "non_target"
MODEL_PATH = "speaker_recognition_model.pth"

# Load the pre-trained model for embedding extraction
spk_rec_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                                savedir="pretrained_models/spkrec-ecapa-voxceleb")

class SpeakerRecognitionModel(nn.Module):
    def __init__(self, input_size):
        super(SpeakerRecognitionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Adjusted to the input size from embeddings
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)  # Use dropout to prevent overfitting
        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x


def get_embedding(audio_path):
    """Get speaker embedding from the audio."""
    signal, fs = librosa.load(audio_path, sr=None)
    signal = torch.tensor(np.expand_dims(signal, axis=0))
    embeddings = spk_rec_model.encode_batch(signal)
    embeddings = embeddings.squeeze().cpu().detach().numpy()
    print(f"Embedding shape: {embeddings.shape}")  # Print the shape of the embeddings
    return embeddings


def augment_data(signal, sr):
    """Placeholder for data augmentation. You can implement noise addition, time shifting, etc."""
    # For example, here we could add noise:
    noise = np.random.randn(len(signal)) * 0.005  # small noise
    augmented_signal = signal + noise
    return augmented_signal

def collect_embeddings(folder, label, augment=False):
    """Collect embeddings from audio files in the given folder."""
    embeddings = []
    labels = []
    for file_name in os.listdir(folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder, file_name)
            embedding = get_embedding(file_path)
            
            # Augmentation logic: You could augment and extract embedding from augmented data
            if augment:
                signal, fs = librosa.load(file_path, sr=None)
                augmented_signal = augment_data(signal, fs)
                augmented_embedding = spk_rec_model.encode_batch(torch.tensor(np.expand_dims(augmented_signal, axis=0)))
                augmented_embedding = augmented_embedding.squeeze().cpu().detach().numpy()
                embeddings.append(augmented_embedding)
                labels.append(label)
            
            embeddings.append(embedding)
            labels.append(label)
    return embeddings, labels

def prepare_data():
    """Prepare data for training the model."""
    authorized_embeddings, authorized_labels = collect_embeddings(AUTHORIZED_FOLDER, 1, augment=True)
    unauthorized_embeddings, unauthorized_labels = collect_embeddings(UNAUTHORIZED_FOLDER, 0, augment=True)
    
    X = authorized_embeddings + unauthorized_embeddings
    y = authorized_labels + unauthorized_labels
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    print("Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    
    print("Initializing model...")
    input_size = len(X_train[0])
    model = SpeakerRecognitionModel(input_size)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # L2 regularization
    early_stopping_patience = 10
    
    print("Starting training...")
    num_epochs = 100
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(len(X_train)):
            optimizer.zero_grad()
            output = model(torch.tensor(X_train[i], dtype=torch.float32))
            loss = criterion(output, torch.tensor([y_train[i]], dtype=torch.float32))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Early stopping condition
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break
    
    print("Training completed. Best model saved.")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        for i in range(len(X_test)):
            output = model(torch.tensor(X_test[i], dtype=torch.float32))
            pred = (output > 0.5).float()
            correct += (pred == y_test[i]).sum().item()
        accuracy = correct / len(X_test)
        print(f"Test Accuracy: {accuracy:.4f}")
    
    return model

def recognize_speaker(audio_path, model):
    """Recognize if the audio is from an authorized speaker."""
    embedding = get_embedding(audio_path)
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(embedding, dtype=torch.float32))
        probability = output.item()
    return probability > 0.8, probability

def main():
    if not os.path.exists(MODEL_PATH):
        print("Training new model...")
        model = train_model()
    else:
        print("Loading existing model...")
        # Ignore existing model if architecture changed
        input_size = len(get_embedding(os.path.join(AUTHORIZED_FOLDER, os.listdir(AUTHORIZED_FOLDER)[0])))
        model = SpeakerRecognitionModel(input_size)
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
        except RuntimeError as e:
            print(f"Model loading failed due to architecture mismatch: {e}")
            print("Training a new model from scratch...")
            model = train_model()

    # Test the model
    test_audio_path = "kri.wav"
    is_authorized, probability = recognize_speaker(test_audio_path, model)
    if is_authorized:
        print(f"Speaker identified as authorized with probability: {probability:.4f}")
    else:
        print(f"Speaker identified as unauthorized. Probability of being authorized: {probability:.4f}")


if __name__ == "__main__":
    main()
