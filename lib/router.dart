import 'package:go_router/go_router.dart';
import 'screens/home_screen.dart';
import 'screens/enroll_screen.dart';
import 'screens/authenticate_screen.dart';
import 'screens/pipeline_screen.dart';
import 'screens/screens.dart';

final appRouter = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(path: '/',             builder: (context, state) => const HomeScreen()),
    GoRoute(path: '/enroll',       builder: (context, state) => const EnrollScreen()),
    GoRoute(path: '/authenticate', builder: (context, state) => const AuthenticateScreen()),
    GoRoute(path: '/verify',       builder: (context, state) => const VerifyScreen()),
    GoRoute(path: '/pipeline',     builder: (context, state) => const PipelineScreen()),
    GoRoute(path: '/history',      builder: (context, state) => const HistoryScreen()),
    GoRoute(path: '/settings',     builder: (context, state) => const SettingsScreen()),
    GoRoute(path: '/qc',           builder: (context, state) => const QcScreen()),
  ],
);
