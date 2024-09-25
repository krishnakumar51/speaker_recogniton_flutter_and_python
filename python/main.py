from flask import Flask, request, jsonify
import os
import uuid
import torch
import traceback
from flask_cors import CORS
from imp import VariableLengthSpeakerVerificationModel, train_model, verify_speaker, get_embedding, MODEL_PATH, SCALER_PATH
import psutil

app = Flask(__name__)
CORS(app)

AUTHORIZED_USER_FOLDER = "authenticated_user"
TEMP_AUDIO_FOLDER = "temp_audio"

# Create necessary directories
os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)
os.makedirs(AUTHORIZED_USER_FOLDER, exist_ok=True)

# Global variables to store the model and scaler
model = None
scaler = None

@app.route('/resources', methods=['GET'])
def resources():
    """Return current resource utilization."""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)

    return jsonify({
        'cpu_usage': cpu,
        'memory_usage': {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used,
            'free': memory.free,
        }
    }), 200



# Function to load the model and scaler once at server startup
def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("Loading existing speaker verification model...")
        input_size = len(get_embedding(os.path.join(AUTHORIZED_USER_FOLDER, os.listdir(AUTHORIZED_USER_FOLDER)[0])))
        model = VariableLengthSpeakerVerificationModel(input_size)
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            scaler = torch.load(SCALER_PATH)
            print("Model and scaler loaded successfully.")
        except RuntimeError as e:
            print(f"Error loading model: {e}. You may need to retrain the model.")
            model = None
            scaler = None
    else:
        print("Model or scaler not found. Please train the model first.")

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200

# New route to check model status
@app.route('/model_status', methods=['GET'])
def model_status():
    if model is not None and scaler is not None:
        return jsonify({'message': 'Model is already trained and loaded.'}), 200
    else:
        return jsonify({'message': 'No trained model found. Please train the model first.'}), 404

@app.route('/upload_sample', methods=['POST'])
def upload_sample():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    unique_filename = str(uuid.uuid4()) + '.wav'
    audio_path = os.path.join(AUTHORIZED_USER_FOLDER, unique_filename)
    audio_file.save(audio_path)
    print(f"Audio sample saved at {audio_path}")

    return jsonify({'message': 'Audio sample uploaded successfully'}), 200

@app.route('/train_model', methods=['GET', 'POST'])
def train_model_endpoint():
    global model
    try:
        if model is not None:  # Check if the model is already trained
            return jsonify({'message': 'Model is already trained. Please retrain if necessary.'}), 200
        
        print("Starting model training...")
        model = train_model()
        if model:
            load_model()  # Reload the model and scaler after training
            return jsonify({'message': 'Training completed successfully!'}), 200
        else:
            return jsonify({'message': 'Training failed.'}), 500
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'error': f'Error during training: {str(e)}'}), 500

@app.route('/recognize', methods=['POST'])
def recognize_speaker_endpoint():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    unique_filename = str(uuid.uuid4()) + '.wav'
    temp_path = os.path.join(TEMP_AUDIO_FOLDER, unique_filename)
    audio_file.save(temp_path)
    print(f"Audio file saved at {temp_path}")

    if model is None or scaler is None:  # Check if the model is loaded
        print("Model or scaler is not loaded, cannot proceed with recognition.")
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 400

    try:
        print(f"Processing audio file for recognition: {temp_path}")
        is_authorized, probability = verify_speaker(temp_path, model, scaler)

        # Clean up temporary audio file
        os.remove(temp_path)
        print(f"Temporary file {temp_path} deleted.")

        if is_authorized:
            return jsonify({'message': 'Speaker identified as authorized', 'probability': probability}), 200
        else:
            return jsonify({'message': 'Speaker identified as unauthorized', 'probability': probability}), 200
    except Exception as e:
        print(f"Error during recognition: {str(e)}")  # Enhanced logging for the error
        traceback.print_exc()  # Print the full stack trace
        return jsonify({'error': f'Error during recognition: {str(e)}'}), 500

# Load model once at startup
load_model()

# Disable auto-reloading to avoid unnecessary restarts
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')
