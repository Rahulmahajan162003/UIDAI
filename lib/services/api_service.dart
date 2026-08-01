import 'dart:io';
import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ApiService {
  static const String _defaultUrl = 'http://35.255.138.39:5002';
  static const String _prefKey    = 'server_url_v2';

  static String baseUrl = _defaultUrl;

  static final Dio _dio = Dio(BaseOptions(
    baseUrl: _defaultUrl,
    connectTimeout: const Duration(seconds: 30),
    receiveTimeout: const Duration(seconds: 90),
  ));

  /// Call once at app startup — loads persisted URL from SharedPreferences.
  static Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString(_prefKey);
    if (saved != null && saved.isNotEmpty) {
      baseUrl = saved;
      _dio.options.baseUrl = saved;
    }
  }

  /// Updates the in-memory URL and persists it so it survives restarts.
  static Future<void> setBaseUrl(String url) async {
    baseUrl = url;
    _dio.options.baseUrl = url;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefKey, url);
  }

  static Future<Map<String, dynamic>> healthCheck() async {
    final r = await _dio.get('/health');
    return r.data;
  }

  static Future<Map<String, dynamic>> qualityCheck(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/quality_check', data: fd,
          options: Options(receiveTimeout: const Duration(seconds: 5)));
      return r.data;
    } on DioException {
      return {};
    }
  }

  static Future<Map<String, dynamic>> enroll({
    required String name,
    required String uid,
    required String batch,
    required File image,
  }) async {
    final fd = FormData.fromMap({
      'name': name, 'uid': uid, 'batch': batch,
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/enroll', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }

  static Future<Map<String, dynamic>> authenticate({
    required String batch,
    required File image,
  }) async {
    final fd = FormData.fromMap({
      'batch': batch,
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/authenticate', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }

  static Future<Map<String, dynamic>> verify({
    required String uid,
    required String batch,
    required File image,
  }) async {
    final fd = FormData.fromMap({
      'uid': uid, 'batch': batch,
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/verify', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }

  static Future<Map<String, dynamic>> process(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/process', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }

  static Future<List<dynamic>> getUsers({String batch = ''}) async {
    final r = await _dio.get('/users', queryParameters: {'batch': batch});
    return r.data['users'] ?? [];
  }

  static Future<List<dynamic>> getHistory({String batch = ''}) async {
    final r = await _dio.get('/history', queryParameters: {'batch': batch});
    return r.data['history'] ?? [];
  }

  static Future<Map<String, dynamic>> readiness(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/readiness', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }

  static Future<Map<String, dynamic>> checkRoi(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/check_roi', data: fd,
          options: Options(receiveTimeout: const Duration(seconds: 5)));
      return r.data;
    } on DioException {
      return {};
    }
  }

  static Future<Map<String, dynamic>> livenessGesture({
    required File image,
    required int expectedCount,
  }) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
      'expected_count': expectedCount.toString(),
    });
    try {
      final r = await _dio.post('/liveness_gesture', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }
}
