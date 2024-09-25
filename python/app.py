from flask import Flask, request, jsonify
import os
import uuid
from speaker_recognition import (
    register_authorized_speaker,
    is_authorized_speaker,
    train_speaker_model,
    # augment_data,
    recognize_speaker_with_wake_word,
    # NOISE_FOLDER
)
from flask_cors import CORS  
import traceback
import numpy as np

app = Flask(__name__)
CORS(app)

AUTHORIZED_USER_FOLDER = "authenticated_user"
TEMP_AUDIO_FOLDER = "temp_audio"
SPEAKER = "speaker"
EMBEDDINGS_FILE_PATH = os.path.join(SPEAKER, "authorized_embedding_avg.npy")

# Create necessary directories
os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)
os.makedirs(SPEAKER, exist_ok=True)
# os.makedirs(NOISE_FOLDER, exist_ok=True)
os.makedirs(AUTHORIZED_USER_FOLDER, exist_ok=True)

# Load authorized embedding if it exists
authorized_embedding_avg = None
if os.path.exists(EMBEDDINGS_FILE_PATH):
    authorized_embedding_avg = np.load(EMBEDDINGS_FILE_PATH)


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200

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

    # Augment data for noise robustness
    # augment_data(audio_path, AUTHORIZED_USER_FOLDER)

    return jsonify({'message': 'Audio sample uploaded successfully'}), 200

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    global authorized_embedding_avg
    try:
        print("Starting model training...")
        
        wav_files = [f for f in os.listdir(AUTHORIZED_USER_FOLDER) if f.endswith('.wav')]
        file_count = len(wav_files)
        
        if file_count == 0:
            return jsonify({'message': 'No audio samples found. Please upload samples before training.'}), 400
        
        print(f"Found {file_count} audio samples for training.")
        
        authorized_embedding_avg = train_speaker_model(AUTHORIZED_USER_FOLDER)
        
        if authorized_embedding_avg is not None:
            print("Training completed successfully.")
            return jsonify({'message': f'Training completed successfully with {file_count} samples!'}), 200
        else:
            print("Training failed: authorized_embedding_avg is None.")
            return jsonify({'message': 'Training failed. Check server logs for details.'}), 500
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'error': f'Error during training: {str(e)}'}), 500




@app.route('/recognize', methods=['POST'])
def recognize_speaker():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if audio_file and authorized_embedding_avg is not None:
        unique_filename = str(uuid.uuid4()) + '.wav'
        temp_path = os.path.join(TEMP_AUDIO_FOLDER, unique_filename)
        audio_file.save(temp_path)
        print(f"Audio file saved at {temp_path}")

        try:
            # Call the recognize_speaker_with_wake_word function from speaker_recognition.py
            recognition_message, similarity = recognize_speaker_with_wake_word(temp_path, authorized_embedding_avg)

            # Clean up temporary audio file
            os.remove(temp_path)
            print(f"Temporary file {temp_path} deleted.")

            response = {
                'message': recognition_message,
                'similarity': float(similarity) if similarity is not None else None
            }

            return jsonify(response), 200

        except Exception as e:
            print(f"Error during recognition: {str(e)}")
            print("Exception type:", type(e).__name__)
            print("Exception args:", e.args)
            print("Traceback:")
            traceback.print_exc(file=sys.stdout)
            
            error_info = {
                'error': f'Error during recognition: {str(e)}',
                'exception_type': type(e).__name__,
                'exception_args': e.args,
                'traceback': traceback.format_exc()
            }
            return jsonify(error_info), 500
    else:
        return jsonify({'error': 'Authorized speaker not registered or training not completed'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
