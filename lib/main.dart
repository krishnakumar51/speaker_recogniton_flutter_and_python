import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'services/api_service.dart';
import 'package:device_info_plus/device_info_plus.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Speaker Recognition Demo',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: SpeakerRecognitionDemo(),
    );
  }
}

class SpeakerRecognitionDemo extends StatefulWidget {
  @override
  _SpeakerRecognitionDemoState createState() => _SpeakerRecognitionDemoState();
}

class _SpeakerRecognitionDemoState extends State<SpeakerRecognitionDemo> {
  bool _isPinging = false;
  String serverResponse = "Tap the button to ping the server";
  FlutterSoundRecorder? _recorder;
  bool _isRecording = false;
  String _result = '';
  String _errorMessage = '';
  late Directory directory;
  String filePath = "";
  int sampleCount = 0; // Tracks the number of uploaded samples
  final int maxSamples = 00; // Limit to 10 samples
  bool isModelTrained = false; // Indicates if the model has been trained
  String modelStatus = '';

  @override
  void initState() {
    super.initState();
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    directory =
        await getApplicationDocumentsDirectory(); // Use app documents directory
    _recorder = FlutterSoundRecorder();

    bool micPermissionGranted = await _requestMicrophonePermission();
    bool storagePermissionGranted = await _requestStoragePermission();
    print(
        "micPermissionGranted $micPermissionGranted storagePermissionGranted $storagePermissionGranted");
    if (!micPermissionGranted || !storagePermissionGranted) {
      setState(() {
        _errorMessage = 'Microphone or storage permission not granted';
      });
      return;
    }

    try {
      await _recorder!.openRecorder();
      print('Recorder opened successfully');
    } catch (e) {
      setState(() {
        _errorMessage = 'Error opening recorder: $e';
      });
      print('Error opening recorder: $e');
    }

    // Optionally, fetch the current sample count from the server
    // You may need to implement an API endpoint to get the current sample count
  }

  Future<void> _checkModelStatus() async {
    setState(() {
      _result = 'Checking model status...';
    });
    try {
      String response = await ApiService.checkModelStatus();
      setState(() {
        modelStatus = response;
      });
    } catch (e) {
      setState(() {
        modelStatus = 'Error: $e';
      });
    }
  }

  Future<bool> _requestMicrophonePermission() async {
    PermissionStatus status = await Permission.microphone.request();
    return status == PermissionStatus.granted;
  }

  Future<bool> _requestStoragePermission() async {
    PermissionStatus status;
    DeviceInfoPlugin deviceInfo = DeviceInfoPlugin();
    AndroidDeviceInfo androidInfo = await deviceInfo.androidInfo;
    if (Platform.isAndroid && androidInfo.version.sdkInt > 13) {
      status = await Permission.accessMediaLocation.request();
    } else {
      status = await Permission.storage.request();
    }
    return status == PermissionStatus.granted;
  }

  Future<void> _startRecording() async {
    if (_recorder == null || !_recorder!.isStopped) {
      return;
    }

    try {
      filePath =
          '${directory.path}/audio_${sampleCount + 1}.wav'; // Name file by sample count
      await _recorder!.startRecorder(
        toFile: filePath,
        codec: Codec.pcm16WAV,
      );
      setState(() {
        _isRecording = true;
        _errorMessage = '';
      });
      print('Recording started. Saving to: $filePath');
    } catch (e) {
      setState(() {
        _errorMessage = 'Error starting recording: $e';
      });
      print('Error starting recording: $e');
    }
  }

  Future<void> _stopRecording() async {
    try {
      await _recorder!.stopRecorder();
      setState(() {
        _isRecording = false;
        _errorMessage = '';
      });
      print('Recording stopped. File saved at: $filePath');

      if (filePath.isNotEmpty && File(filePath).existsSync()) {
        await _sendAudioToServer(filePath);
      } else {
        print('File not found: $filePath');
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error stopping recording: $e';
      });
      print('Error stopping recording: $e');
    }
  }

  Future<void> _sendAudioToServer(String audioPath) async {
    setState(() {
      _result = 'Uploading sample...';
    });
    try {
      String response = await ApiService.uploadSample(File(audioPath));
      setState(() {
        sampleCount++; // Increment the sample count after successful upload
        if (sampleCount >= maxSamples) {
          _result = '10 samples collected! You can now train the model.';
        } else {
          _result = response + '\nSamples collected: $sampleCount/$maxSamples';
        }
      });
      print(response);
    } catch (e) {
      setState(() {
        _result = 'Error: $e';
      });
      print('Error uploading sample: $e');
    }
  }

  Future<void> _sendAudioForTraining() async {
    if (sampleCount < maxSamples) {
      setState(() {
        _result = 'Need 10 samples to start training. Collect more samples.';
      });
      return;
    }

    setState(() {
      _result = 'Training model...';
    });

    try {
      String response = await ApiService.trainSpeakerModel();
      setState(() {
        _result = response;
        isModelTrained = true;
      });
      print(response);
    } catch (e) {
      setState(() {
        _result = 'Error: $e';
      });
      print('Error triggering model training: $e');
    }
  }

  Future<void> _testSpeaker() async {
    setState(() {
      _result = 'Recording for recognition...';
    });

    await _startTestRecording();
  }

  Future<void> _startTestRecording() async {
    if (_recorder == null || !_recorder!.isStopped) {
      return;
    }

    try {
      String testFilePath = '${directory.path}/test_audio.wav';
      await _recorder!.startRecorder(
        toFile: testFilePath,
        codec: Codec.pcm16WAV,
      );
      setState(() {
        _isRecording = true;
        _errorMessage = '';
      });
      print('Test recording started. Saving to: $testFilePath');

      // Automatically stop recording after a certain duration (e.g., 3 seconds)
      await Future.delayed(Duration(seconds: 3));
      await _recorder!.stopRecorder();
      setState(() {
        _isRecording = false;
      });
      print('Test recording stopped. File saved at: $testFilePath');

      if (testFilePath.isNotEmpty && File(testFilePath).existsSync()) {
        await _sendTestAudioToServer(testFilePath);
      } else {
        print('Test file not found: $testFilePath');
        setState(() {
          _result = 'Test audio file not found.';
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Error during test recording: $e';
      });
      print('Error during test recording: $e');
    }
  }

  Future<void> _sendTestAudioToServer(String audioPath) async {
    setState(() {
      _result = 'Recognizing speaker...';
    });
    try {
      String response = await ApiService.recognizeSpeaker(File(audioPath));
      setState(() {
        _result = response;
      });
      print('Recognition result: $response');
    } catch (e) {
      setState(() {
        _result = 'Error: $e';
      });
      print('Error during speaker recognition: $e');
    }
  }

  void pingServer() async {
    setState(() {
      _isPinging = true; // Set to true when starting the ping
    });
    try {
      String response = await ApiService.pingServer();
      setState(() {
        serverResponse = response;
      });
    } catch (e) {
      setState(() {
        serverResponse = 'Error pinging server: $e';
      });
    } finally {
      setState(() {
        _isPinging = false; // Reset the state variable after completion
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Speaker Recognition Demo'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0), // Added padding for better UI
        child: Center(
          child: SingleChildScrollView(
            // Added scroll view to handle overflow
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                // Model Status Section
                Text(
                  'Model Status',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 10),
                ElevatedButton(
                  onPressed: _checkModelStatus, // Button to check model status
                  child: Text('Check Model Status'),
                ),
                SizedBox(height: 10),
                Text(
                  modelStatus.isNotEmpty
                      ? modelStatus
                      : 'No status checked yet.',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.normal),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 20),
                // Recording Samples Section
                Text(
                  'Collect Samples',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 10),
                ElevatedButton(
                  onPressed: _isRecording || sampleCount >= maxSamples
                      ? null
                      : _startRecording,
                  child: Text('Start Recording Sample'),
                ),
                SizedBox(height: 10),
                ElevatedButton(
                  onPressed: _isRecording ? _stopRecording : null,
                  child: Text('Stop Recording Sample'),
                ),
                SizedBox(height: 10),
                Text(
                  'Samples collected: $sampleCount/$maxSamples',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.normal),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 20),

                // Training Section
                Text(
                  'Training',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 10),
                ElevatedButton(
                  onPressed: (sampleCount >= maxSamples && !isModelTrained)
                      ? _sendAudioForTraining
                      : null,
                  child: Text('Train Model'),
                ),
                SizedBox(height: 10),
                Text(
                  isModelTrained
                      ? 'Model trained successfully!'
                      : 'Model not trained yet.',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.normal,
                    color: isModelTrained ? Colors.green : Colors.orange,
                  ),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 20),

                // Testing Section
                Text(
                  'Test Model',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 10),
                ElevatedButton(
                  onPressed:
                      isModelTrained && !_isRecording ? _testSpeaker : null,
                  child: Text('Test Speaker Recognition'),
                ),
                SizedBox(height: 10),
                Text(
                  'Press the button to record a 3-second test audio for recognition.',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.normal),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _isPinging
                      ? null
                      : pingServer, // Disable button while pinging
                  child: _isPinging
                      ? CircularProgressIndicator()
                      : Text('Ping Server'),
                ),

                // Result Display
                Divider(),
                SizedBox(height: 10),
                Text(
                  'Result:',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 10),
                Text(
                  _result,
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.normal),
                  textAlign: TextAlign.center,
                ),
                if (_errorMessage.isNotEmpty) // Display error message if exists
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Text(
                      _errorMessage,
                      style: TextStyle(color: Colors.red),
                      textAlign: TextAlign.center,
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _recorder?.closeRecorder();
    super.dispose();
  }
}
