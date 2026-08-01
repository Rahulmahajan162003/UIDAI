import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'theme/app_theme.dart';
import 'router.dart';
import 'screens/splash_screen.dart';
import 'services/api_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Load persisted server URL before anything tries to call the API.
  await ApiService.init();
  // Request camera permission upfront so it never interrupts the capture flow.
  // On first install this shows the system dialog here, before any camera
  // widget is visible — avoids the null-crash that occurs when permission is
  // granted mid-initialization.
  await Permission.camera.request();
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp, DeviceOrientation.portraitDown,
  ]);
  SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
    statusBarColor: Colors.transparent,
    statusBarIconBrightness: Brightness.dark,
  ));
  runApp(const YellowSenseApp());
}

class YellowSenseApp extends StatefulWidget {
  const YellowSenseApp({super.key});
  @override
  State<YellowSenseApp> createState() => _YellowSenseAppState();
}

class _YellowSenseAppState extends State<YellowSenseApp> {
  bool _showSplash = true;

  @override
  Widget build(BuildContext context) {
    if (_showSplash) {
      return MaterialApp(
        title: 'YellowSense UIDAI',
        debugShowCheckedModeBanner: false,
        theme: YS.theme,
        home: SplashScreen(onDone: () => setState(() => _showSplash = false)),
      );
    }
    return MaterialApp.router(
      title: 'YellowSense UIDAI',
      debugShowCheckedModeBanner: false,
      theme: YS.theme,
      routerConfig: appRouter,
    );
  }
}
