import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:audioplayers/audioplayers.dart';

void main() {
  runApp(const MyApp());
}

// Replace with your local IP if needed
const String baseUrl = 'http://192.0.0.2:8000'; 

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Agro AI App',
      theme: ThemeData(primarySwatch: Colors.green),
      home: const MyHomePage(title: 'Crop Disease Analyzer'),
      debugShowCheckedModeBanner: false,
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  String? _result;
  String? _language = "en";
  bool _isLoading = false;
  final AudioPlayer _audioPlayer = AudioPlayer();

  Future<void> _pickImage() async {
    final pickedFile = await ImagePicker().pickImage(source: ImageSource.gallery);
    setState(() {
      _image = pickedFile != null ? File(pickedFile.path) : null;
      _result = null;
    });
  }

  Future<void> _analyzeImage() async {
    if (_image == null || _language == null) {
      setState(() {
        _result = 'Please select an image and language.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _result = null;
    });

    try {
      final uri = Uri.parse('$baseUrl/analyze/');
      final request = http.MultipartRequest('POST', uri)
        ..fields['lang'] = _language!
        ..files.add(await http.MultipartFile.fromPath('image', _image!.path));

      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final message = data['message'] ?? 'Disease detected';

        setState(() {
          _result = message;
        });

        if (data.containsKey('audio_file')) {
          final audioUrl = '$baseUrl/audio/${data['audio_file']}';
          await _audioPlayer.play(UrlSource(audioUrl));
        }
      } else {
        setState(() {
          _result = 'Server error: ${response.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        _result = 'Error occurred: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.title)),
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              if (_image != null)
                ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: Image.file(_image!, height: 200),
                ),
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: _pickImage,
                icon: const Icon(Icons.image),
                label: const Text('Pick Image'),
              ),
              const SizedBox(height: 10),
              DropdownButton<String>(
                value: _language,
                isExpanded: true,
                items: const [
                  DropdownMenuItem(value: 'en', child: Text('English')),
                  DropdownMenuItem(value: 'hi', child: Text('Hindi')),
                  DropdownMenuItem(value: 'mr', child: Text('Marathi')),
                ],
                onChanged: (value) {
                  setState(() => _language = value);
                },
              ),
              const SizedBox(height: 10),
              ElevatedButton.icon(
                onPressed: _isLoading ? null : _analyzeImage,
                icon: const Icon(Icons.search),
                label: const Text('Analyze'),
              ),
              const SizedBox(height: 20),
              if (_isLoading)
                const CircularProgressIndicator()
              else if (_result != null)
                Text(
                  _result!,
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
                  textAlign: TextAlign.center,
                ),
            ],
          ),
        ),
      ),
    );
  }
}
