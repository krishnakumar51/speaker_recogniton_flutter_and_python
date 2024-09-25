import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl =
      'http://192.168.1.38:5000'; // Replace with your actual server IP

  // New function to check model status
  static Future<String> checkModelStatus() async {
    var uri = Uri.parse('$baseUrl/model_status');
    print('Checking model status at: $uri');

    try {
      var response = await http.get(uri); // GET request to check model status

      print('Response status: ${response.statusCode}');
      print('Response body: ${response.body}');

      if (response.statusCode == 200) {
        var responseData = jsonDecode(response.body);
        return responseData['message'];
      } else if (response.statusCode == 404) {
        return 'No trained model found. Please train the model first.';
      } else {
        throw Exception('Failed to check model status: ${response.statusCode}');
      }
    } catch (e) {
      print('Error checking model status: $e');
      throw Exception('Failed to check model status: $e');
    }
  }

  // Function to upload a single audio sample to the server
  static Future<String> uploadSample(File audioFile) async {
    var uri = Uri.parse('$baseUrl/upload_sample');
    print('Uploading sample to: $uri');
    try {
      var request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('audio', audioFile.path));
      print('Sending file: ${audioFile.path}');
      var streamedResponse = await request.send();
      print("streamedResponse $streamedResponse");
      var response = await http.Response.fromStream(streamedResponse);

      print('Response status: ${response.statusCode}');
      print('Response body: ${response.body}');

      if (response.statusCode == 200) {
        return 'Sample uploaded successfully';
      } else {
        throw Exception('Failed to upload sample: ${response.statusCode}');
      }
    } catch (e) {
      print('Error uploading sample: $e');
      throw Exception('Failed to upload sample: $e');
    }
  }

  // Function to send a request to start model training on the server
  static Future<String> trainSpeakerModel() async {
    var uri = Uri.parse('$baseUrl/train_model');
    print('Requesting training at: $uri');

    try {
      var response = await http.post(uri); // POST request to train the model

      print('Response status: ${response.statusCode}');
      print('Response body: ${response.body}');

      if (response.statusCode == 200) {
        return 'Model training completed successfully';
      } else {
        throw Exception('Failed to train model: ${response.statusCode}');
      }
    } catch (e) {
      print('Error triggering model training: $e');
      throw Exception('Failed to trigger model training: $e');
    }
  }

  static Future<String> pingServer() async {
    var uri = Uri.parse('$baseUrl/ping');
    print(uri.toString());
    try {
      // Send the GET request to the server
      var response = await http.get(uri);

      if (response.statusCode == 200) {
        // Decode the JSON response
        var responseData = jsonDecode(response.body);
        print(
            'Server Response: ${responseData['message']}'); // Print the response in console
        return responseData['message'] ??
            'No message in response'; // Handle potential null
      } else {
        print('Failed to ping server. Status code: ${response.statusCode}');
        return 'Failed to ping server';
      }
    } catch (e) {
      print('Error occurred while pinging the server: $e');
      return 'Error: $e';
    }
  }

  // Function to test the model with a new audio file
  static Future<String> recognizeSpeaker(File audioFile) async {
    var uri = Uri.parse('$baseUrl/recognize');
    print('Base URL: $baseUrl');
    print('Audio file path: ${audioFile.path}');

    if (!await audioFile.exists()) {
      throw Exception('Audio file does not exist at ${audioFile.path}');
    }

    var request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('audio', audioFile.path));

    try {
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return response.body;
      } else {
        print('Response body: ${response.body}');
        throw Exception('Failed to recognize speaker: ${response.statusCode}');
      }
    } catch (e) {
      print('Error during recognition: $e');
      throw Exception('Error during recognition: $e');
    }
  }
}
