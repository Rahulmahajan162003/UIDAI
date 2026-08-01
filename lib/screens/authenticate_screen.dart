import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import '../theme/app_theme.dart';
import '../widgets/fingerprint_camera_widget.dart';
import '../services/api_service.dart';
import 'screens.dart' show ysShimmer, ysOfflineCard, regionDropdown;

class AuthenticateScreen extends StatefulWidget {
  const AuthenticateScreen({super.key});
  @override
  State<AuthenticateScreen> createState() => _AuthenticateScreenState();
}

class _AuthenticateScreenState extends State<AuthenticateScreen> {
  String? _region;
  bool _loading = false;
  Map<String, dynamic>? _result;

  Future<void> _authenticate(File image) async {
    if (_region == null) {
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Select a region', style: YS.label(13)),
              backgroundColor: YS.red));
      return;
    }
    setState(() { _loading = true; _result = null; });
    try {
      final r = await ApiService.authenticate(batch: _region!, image: image);
      setState(() { _result = r; _loading = false; });
      if (r['success'] == true) { HapticFeedback.heavyImpact(); }
      else { HapticFeedback.vibrate(); }
    } catch (e) {
      final isOffline = e.toString().contains('SocketException') ||
          e.toString().contains('Connection refused') ||
          e.toString().contains('timed out');
      setState(() {
        _result = {'success': false, 'error': e.toString(), 'offline': isOffline};
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: YS.bg,
      appBar: AppBar(
        leading: IconButton(icon: const Icon(Icons.arrow_back_rounded), onPressed: () => context.go('/')),
        title: Text('Authenticate', style: YS.label(17, w: FontWeight.w700)),
        bottom: PreferredSize(preferredSize: const Size.fromHeight(1),
            child: Container(height: 1, color: YS.stroke)),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(color: YS.greenBg, borderRadius: BorderRadius.circular(12),
                border: Border.all(color: YS.green.withValues(alpha: 0.3))),
            child: Row(children: [
              const Icon(Icons.info_outline_rounded, color: YS.green, size: 16),
              const SizedBox(width: 8),
              Expanded(child: Text('Matches against all enrolled users in the batch',
                  style: YS.label(12, color: YS.green))),
            ]),
          ),
          const SizedBox(height: 20),
          regionDropdown(
            value: _region,
            onChanged: (v) => setState(() => _region = v),
          ),
          const SizedBox(height: 24),
          Text('FINGERPRINT CAPTURE', style: YS.label(11, color: YS.inkLight, w: FontWeight.w700)
              .copyWith(letterSpacing: 1.8)),
          const SizedBox(height: 10),
          FingerprintCameraWidget(onImageCaptured: _authenticate, disabled: _loading),
          const SizedBox(height: 20),
          if (_loading) ...[ysShimmer(height: 80), const SizedBox(height: 8), ysShimmer(height: 60)],
          if (!_loading && _result != null) ...[const SizedBox(height: 16),
            if (_result!['offline'] == true)
              ysOfflineCard(() => setState(() => _result = null))
            else
              _resultCard(_result!)
          ],
          const SizedBox(height: 40),
        ]),
      ),
    );
  }

  Widget _resultCard(Map<String, dynamic> r) {
    final ok    = r['success'] == true;
    final color = ok ? YS.green : YS.red;
    final bg    = ok ? YS.greenBg : YS.redBg;
    String msg  = ok ? '' : (r['message'] ?? r['error'] ?? 'Not recognized');
    if (r['quality_failed'] == true) msg = r['guidance'] ?? 'Quality check failed';
    if (r['spoof_detected'] == true) msg = 'Spoof detected';
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withValues(alpha: 0.3))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(ok ? Icons.verified_rounded : Icons.cancel_rounded, color: color, size: 24),
          const SizedBox(width: 10),
          Text(ok ? 'AUTHENTICATED' : 'NOT RECOGNIZED',
              style: YS.display(16, color: color, w: FontWeight.w800)),
        ]),
        const SizedBox(height: 14),
        if (ok) ...[
          _row('Name', r['name'] ?? '—'),
          _row('UID',  r['uid']  ?? '—'),
          _row('Confidence', '${((r['confidence'] ?? 0) * 100).toStringAsFixed(1)}%'),
        ] else
          Text(msg, style: YS.label(13, color: YS.inkMid)),
      ]),
    );
  }

  Widget _row(String k, String v) => Padding(
    padding: const EdgeInsets.only(bottom: 6),
    child: Row(children: [
      Text('$k: ', style: YS.label(13, color: YS.inkMid)),
      Text(v, style: YS.label(13, w: FontWeight.w700)),
    ]),
  );
}
