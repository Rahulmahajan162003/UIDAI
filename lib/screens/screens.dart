import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import 'package:shimmer/shimmer.dart';
import '../theme/app_theme.dart';
import '../widgets/fingerprint_camera_widget.dart';
import '../services/api_service.dart';

// ── India regions (States + UTs) — used as "batch" across all screens ────────

const List<String> kIndiaRegions = [
  // States
  'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
  'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka',
  'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram',
  'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
  'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
  // Union Territories
  'Andaman & Nicobar Islands', 'Chandigarh',
  'Dadra & Nagar Haveli and Daman & Diu', 'Delhi', 'Jammu & Kashmir',
  'Ladakh', 'Lakshadweep', 'Puducherry',
];

/// Shared region / state dropdown used on Enroll, Authenticate and Verify.
Widget regionDropdown({
  required String? value,
  required ValueChanged<String?> onChanged,
}) =>
    DropdownButtonFormField<String>(
      value: value,
      decoration: InputDecoration(
        labelText: 'Region / State',
        hintText: 'Select state or UT',
        prefixIcon: const Icon(Icons.location_on_outlined, size: 18),
        border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
      ),
      isExpanded: true,
      items: kIndiaRegions
          .map((r) => DropdownMenuItem(value: r, child: Text(r)))
          .toList(),
      onChanged: onChanged,
    );



Widget ysShimmer({double height = 80, double? width, double radius = 14}) =>
    Shimmer.fromColors(
      baseColor: YS.stroke,
      highlightColor: YS.cardAlt,
      child: Container(
        width: width,
        height: height,
        decoration: BoxDecoration(
          color: YS.card,
          borderRadius: BorderRadius.circular(radius),
        ),
      ),
    );

Widget ysOfflineCard(VoidCallback onRetry) => Container(
  padding: const EdgeInsets.all(24),
  decoration: BoxDecoration(
    color: YS.card,
    borderRadius: BorderRadius.circular(16),
    border: Border.all(color: YS.stroke),
  ),
  child: Column(children: [
    const Icon(Icons.wifi_off_rounded, size: 40, color: YS.inkLight),
    const SizedBox(height: 12),
    Text('Cannot reach server', style: YS.label(14, color: YS.inkMid, w: FontWeight.w600)),
    const SizedBox(height: 4),
    Text('Check Settings → Server URL', style: YS.label(12, color: YS.inkLight)),
    const SizedBox(height: 16),
    SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: onRetry,
        child: const Text('Retry'),
      ),
    ),
  ]),
);

// ══════════════════════════════════════════════════════════════════════════════
// VERIFY  1:1
// ══════════════════════════════════════════════════════════════════════════════

class VerifyScreen extends StatefulWidget {
  const VerifyScreen({super.key});
  @override
  State<VerifyScreen> createState() => _VerifyScreenState();
}

class _VerifyScreenState extends State<VerifyScreen> {
  final _uidCtrl = TextEditingController();
  String? _region;
  bool _loading = false;
  Map<String, dynamic>? _result;

  @override
  void dispose() { _uidCtrl.dispose(); super.dispose(); }

  Future<void> _verify(File image) async {
    if (_uidCtrl.text.trim().isEmpty || _region == null) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Enter UID and select region', style: YS.label(13)),
          backgroundColor: YS.red));
      return;
    }
    setState(() { _loading = true; _result = null; });
    try {
      final r = await ApiService.verify(
          uid: _uidCtrl.text.trim(), batch: _region!, image: image);
      setState(() { _result = r; _loading = false; });
      if (r['matched'] == true) { HapticFeedback.heavyImpact(); }
      else { HapticFeedback.vibrate(); }
    } catch (e) {
      final msg = e.toString();
      final isOffline = msg.contains('SocketException') ||
          msg.contains('Connection refused') ||
          msg.contains('timed out');
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
        title: Text('Verify Identity', style: YS.label(17, w: FontWeight.w700)),
        bottom: PreferredSize(preferredSize: const Size.fromHeight(1),
            child: Container(height: 1, color: YS.stroke)),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(color: YS.blueBg, borderRadius: BorderRadius.circular(12),
                border: Border.all(color: YS.blue.withValues(alpha: 0.3))),
            child: Row(children: [
              const Icon(Icons.info_outline_rounded, color: YS.blue, size: 16),
              const SizedBox(width: 8),
              Expanded(child: Text('Verifies fingerprint against a specific Aadhaar UID',
                  style: YS.label(12, color: YS.blue))),
            ]),
          ),
          const SizedBox(height: 20),
          regionDropdown(
            value: _region,
            onChanged: (v) => setState(() => _region = v),
          ),
          const SizedBox(height: 12),
          TextField(controller: _uidCtrl, style: YS.label(14),
            decoration: InputDecoration(labelText: 'Aadhaar UID',
                prefixIcon: Icon(Icons.badge_outlined, color: YS.inkLight, size: 18))),
          const SizedBox(height: 24),
          Text('FINGERPRINT CAPTURE', style: YS.label(11, color: YS.inkLight, w: FontWeight.w700)
              .copyWith(letterSpacing: 1.8)),
          const SizedBox(height: 10),
          FingerprintCameraWidget(onImageCaptured: _verify, disabled: _loading),
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
    final ok    = r['success'] == true && r['matched'] == true;
    final color = ok ? YS.green : YS.red;
    final bg    = ok ? YS.greenBg : YS.redBg;
    final score = (r['confidence'] ?? 0.0) as num;
    String msg  = r['error'] ?? '';
    if (r['quality_failed'] == true) msg = r['guidance'] ?? 'Quality failed';
    if (r['spoof_detected'] == true) msg = 'Spoof detected';
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(16),
          border: Border.all(color: color.withValues(alpha: 0.3))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(ok ? Icons.verified_rounded : Icons.cancel_rounded, color: color, size: 24),
          const SizedBox(width: 10),
          Text(ok ? 'IDENTITY VERIFIED' : 'MISMATCH',
              style: YS.display(16, color: color, w: FontWeight.w800)),
        ]),
        const SizedBox(height: 14),
        if (r['success'] == true) ...[
          _row('Name', r['name'] ?? '—'),
          _row('UID',  r['uid']  ?? '—'),
          const SizedBox(height: 10),
          Text('Match Score', style: YS.label(11, color: YS.inkLight, w: FontWeight.w600)
              .copyWith(letterSpacing: 0.5)),
          const SizedBox(height: 6),
          ClipRRect(borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(value: score.toDouble().clamp(0, 1),
                backgroundColor: YS.stroke, color: color, minHeight: 8)),
          const SizedBox(height: 4),
          Text('${(score * 100).toStringAsFixed(1)}%  (threshold 25%)',
              style: YS.label(11, color: YS.inkLight)),
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

// ══════════════════════════════════════════════════════════════════════════════
// QC SCREEN
// ══════════════════════════════════════════════════════════════════════════════

class QcScreen extends StatefulWidget {
  const QcScreen({super.key});
  @override
  State<QcScreen> createState() => _QcScreenState();
}

class _QcScreenState extends State<QcScreen> {
  bool _loading = false;
  Map<String, dynamic>? _result;
  Map<String, dynamic>? _readiness;

  Future<void> _run(File image) async {
    setState(() { _loading = true; _result = null; _readiness = null; });
    try {
      // Run process + readiness in parallel
      final results = await Future.wait([
        ApiService.process(image),
        ApiService.readiness(image),
      ]);
      setState(() {
        _result    = results[0];
        _readiness = results[1];
        _loading   = false;
      });
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
        title: Text('QC Pipeline', style: YS.label(17, w: FontWeight.w700)),
        bottom: PreferredSize(preferredSize: const Size.fromHeight(1),
            child: Container(height: 1, color: YS.stroke)),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          FingerprintCameraWidget(onImageCaptured: _run, disabled: _loading),
          const SizedBox(height: 20),
          if (_loading) ...[ysShimmer(height: 80), const SizedBox(height: 8), ysShimmer(height: 200)],
          if (!_loading && _result?['offline'] == true) ...[const SizedBox(height: 16),
            ysOfflineCard(() => setState(() { _result = null; _readiness = null; }))],
          if (!_loading && _readiness != null) ...[const SizedBox(height: 16), _readinessCard(_readiness!)],
          if (!_loading && _result != null && _result!['offline'] != true) ...[const SizedBox(height: 16), _qcCard(_result!)],
          const SizedBox(height: 40),
        ]),
      ),
    );
  }

  Widget _readinessCard(Map<String, dynamic> r) {
    final score = (r['readiness_score'] as num?)?.toInt() ?? 0;
    final grade = r['grade'] as String? ?? '—';
    final breakdown = r['breakdown'] as Map<String, dynamic>? ?? {};
    final gradeColor = grade == 'Excellent' ? YS.green
        : grade == 'Good'     ? YS.green
        : grade == 'Marginal' ? YS.orange
        : YS.red;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: YS.card, borderRadius: BorderRadius.circular(16),
        border: Border.all(color: YS.stroke), boxShadow: YS.cardShadow,
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          Text('Readiness Score', style: YS.display(16, w: FontWeight.w700)),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
            decoration: BoxDecoration(
              color: gradeColor.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: gradeColor.withValues(alpha: 0.3)),
            ),
            child: Text(grade, style: YS.label(12, color: gradeColor, w: FontWeight.w700)),
          ),
        ]),
        const SizedBox(height: 16),
        Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
          Text('$score', style: YS.display(40, color: gradeColor, w: FontWeight.w800)),
          Padding(padding: const EdgeInsets.only(bottom: 6, left: 4),
            child: Text('/100', style: YS.label(14, color: YS.inkLight))),
        ]),
        const SizedBox(height: 10),
        ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: LinearProgressIndicator(
            value: score / 100.0,
            minHeight: 8,
            backgroundColor: YS.stroke,
            color: gradeColor,
          ),
        ),
        const SizedBox(height: 16),
        Row(children: [
          _miniStat('Blur',       '${breakdown['blur'] ?? '—'}'),
          _miniStat('Brightness', '${breakdown['brightness'] ?? '—'}'),
          _miniStat('Glare',      breakdown['glare'] == true ? 'Yes' : 'No'),
          _miniStat('Minutiae',   '${breakdown['minutiae'] ?? '—'}'),
        ]),
      ]),
    );
  }

  Widget _miniStat(String k, String v) => Expanded(
    child: Container(
      margin: const EdgeInsets.only(right: 8),
      padding: const EdgeInsets.symmetric(vertical: 10),
      decoration: BoxDecoration(
        color: YS.cardAlt, borderRadius: BorderRadius.circular(10)),
      child: Column(children: [
        Text(v, style: YS.label(13, w: FontWeight.w700)),
        const SizedBox(height: 2),
        Text(k, style: YS.label(10, color: YS.inkLight)),
      ]),
    ),
  );

  Widget _qcCard(Map<String, dynamic> r) {
    if (r['success'] != true) {
      return Container(padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(color: YS.redBg, borderRadius: BorderRadius.circular(14),
            border: Border.all(color: YS.red.withValues(alpha: 0.3))),
        child: Text('Error: ${r['error']}', style: YS.label(13, color: YS.red)));
    }
    final qc       = r['quality']  as Map<String, dynamic>? ?? {};
    final liveness = r['liveness'] as Map<String, dynamic>? ?? {};
    final qcPassed = qc['passed'] == true;
    final isLive   = liveness['is_live'] == true;
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(color: YS.card, borderRadius: BorderRadius.circular(16),
          border: Border.all(color: YS.stroke), boxShadow: YS.cardShadow),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text('Pipeline Results', style: YS.display(16, w: FontWeight.w700)),
        const SizedBox(height: 16),
        _qcRow('Quality Gate',   qcPassed ? 'PASSED' : 'FAILED',   qcPassed ? YS.green : YS.red),
        if (!qcPassed && qc['guidance'] != null)
          Padding(padding: const EdgeInsets.only(left: 16, bottom: 8),
            child: Text('→ ${qc['guidance']}', style: YS.label(12, color: YS.orange))),
        _qcRow('Blur Score',  '${qc['blur']?['blur_score'] ?? '—'}',  YS.inkMid),
        _qcRow('Brightness',  '${qc['brightness']?['brightness'] ?? '—'}', YS.inkMid),
        _qcRow('Glare',       qc['glare']?['has_glare'] == true ? 'Detected' : 'None',
            qc['glare']?['has_glare'] == true ? YS.red : YS.green),
        Divider(color: YS.stroke, height: 24),
        _qcRow('Detection',   '${((r['detection_conf'] ?? 0) * 100).toStringAsFixed(1)}%', YS.inkMid),
        _qcRow('Liveness',    isLive ? 'LIVE' : 'SPOOF', isLive ? YS.green : YS.red),
        _qcRow('Liveness Conf', '${((liveness['confidence'] ?? 0) * 100).toStringAsFixed(1)}%', YS.inkMid),
        Divider(color: YS.stroke, height: 24),
        _qcRow('Minutiae',    '${r['minutiae_count'] ?? 0}', YS.amber),
      ]),
    );
  }

  Widget _qcRow(String k, String v, Color vc) => Padding(
    padding: const EdgeInsets.only(bottom: 10),
    child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text(k, style: YS.label(13, color: YS.inkMid)),
      Text(v, style: YS.label(13, color: vc, w: FontWeight.w700)),
    ]),
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// HISTORY
// ══════════════════════════════════════════════════════════════════════════════

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  String? _region;
  List<dynamic> _history = [];
  bool _loading = false;
  bool _offline = false;

  Future<void> _load() async {
    setState(() { _loading = true; _offline = false; });
    try {
      final h = await ApiService.getHistory(batch: _region ?? '');
      setState(() { _history = h; _loading = false; });
    } catch (e) {
      final isOffline = e.toString().contains('SocketException') ||
          e.toString().contains('Connection refused') ||
          e.toString().contains('timed out');
      setState(() { _loading = false; _offline = isOffline; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: YS.bg,
      appBar: AppBar(
        leading: IconButton(icon: const Icon(Icons.arrow_back_rounded), onPressed: () => context.go('/')),
        title: Text('Auth History', style: YS.label(17, w: FontWeight.w700)),
        bottom: PreferredSize(preferredSize: const Size.fromHeight(1),
            child: Container(height: 1, color: YS.stroke)),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(children: [
          Row(children: [
            Expanded(child: regionDropdown(
              value: _region,
              onChanged: (v) => setState(() => _region = v),
            )),
            const SizedBox(width: 12),
            ElevatedButton(onPressed: _load, child: const Text('Load')),
          ]),
          const SizedBox(height: 16),
          if (_loading) ...[const SizedBox(height: 8), ysShimmer(height: 70), const SizedBox(height: 8), ysShimmer(height: 70), const SizedBox(height: 8), ysShimmer(height: 70)],
          const SizedBox(height: 8),
          Expanded(
            child: _offline
                ? ysOfflineCard(_load)
                : _history.isEmpty && !_loading
                    ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
                        const Icon(Icons.history_rounded, size: 48, color: YS.inkFaint),
                        const SizedBox(height: 12),
                        Text('No records yet', style: YS.label(14, color: YS.inkLight)),
                        const SizedBox(height: 4),
                        Text('Tap Load to fetch history', style: YS.label(12, color: YS.inkFaint)),
                      ]))
                : ListView.separated(
                    itemCount: _history.length,
                    separatorBuilder: (_, s) => Divider(color: YS.stroke, height: 1),
                    itemBuilder: (_, i) {
                      final h = _history[i] as Map<String, dynamic>;
                      final conf = ((h['confidence'] ?? 0) * 100).toStringAsFixed(1);
                      return ListTile(
                        contentPadding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
                        leading: Container(width: 40, height: 40,
                          decoration: BoxDecoration(color: YS.amberSoft,
                              borderRadius: BorderRadius.circular(10)),
                          child: const Icon(Icons.fingerprint_rounded, color: YS.amber, size: 20)),
                        title: Text(h['name'] ?? '—', style: YS.label(14, w: FontWeight.w700)),
                        subtitle: Text('${h['uid']} · $conf% confidence',
                            style: YS.label(12, color: YS.inkMid)),
                        trailing: Text(h['timestamp']?.toString().substring(0, 16) ?? '',
                            style: YS.label(10, color: YS.inkLight)),
                      );
                    }),
          ),
        ]),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// SETTINGS
// ══════════════════════════════════════════════════════════════════════════════

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});
  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _urlCtrl =
      TextEditingController(text: ApiService.baseUrl);
  bool _checking = false;
  Map<String, dynamic>? _health;

  Future<void> _save() async {
    await ApiService.setBaseUrl(_urlCtrl.text.trim());
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Saved', style: YS.label(13)),
          backgroundColor: YS.green));
    }
  }

  Future<void> _check() async {
    setState(() { _checking = true; _health = null; });
    try {
      final h = await ApiService.healthCheck();
      setState(() { _health = h; _checking = false; });
    } catch (e) {
      setState(() { _health = {'error': e.toString()}; _checking = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: YS.bg,
      appBar: AppBar(
        leading: IconButton(icon: const Icon(Icons.arrow_back_rounded), onPressed: () => context.go('/')),
        title: Text('Settings', style: YS.label(17, w: FontWeight.w700)),
        bottom: PreferredSize(preferredSize: const Size.fromHeight(1),
            child: Container(height: 1, color: YS.stroke)),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text('SERVER', style: YS.label(11, color: YS.inkLight, w: FontWeight.w700)
              .copyWith(letterSpacing: 1.8)),
          const SizedBox(height: 12),
          TextField(controller: _urlCtrl, style: YS.label(14),
            decoration: InputDecoration(labelText: 'Flask Server URL',
                hintText: 'http://192.168.x.x:5001',
                prefixIcon: Icon(Icons.dns_outlined, color: YS.inkLight, size: 18))),
          const SizedBox(height: 8),
          Text('Physical device: http://<PC-IP>:5001\nEmulator: http://10.0.2.2:5001',
              style: YS.label(11, color: YS.inkLight)),
          const SizedBox(height: 16),
          Row(children: [
            Expanded(child: ElevatedButton(onPressed: _save, child: const Text('Save URL'))),
            const SizedBox(width: 12),
            Expanded(child: OutlinedButton(
              onPressed: _checking ? null : _check,
              child: _checking
                  ? SizedBox(width: 16, height: 16,
                      child: CircularProgressIndicator(color: YS.amber, strokeWidth: 2))
                  : const Text('Check Health'),
            )),
          ]),
          if (_health != null) ...[const SizedBox(height: 16), _healthCard(_health!)],
          const SizedBox(height: 32),
          Text('ABOUT', style: YS.label(11, color: YS.inkLight, w: FontWeight.w700)
              .copyWith(letterSpacing: 1.8)),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: YS.card, borderRadius: BorderRadius.circular(16),
                border: Border.all(color: YS.stroke), boxShadow: YS.cardShadow),
            child: Column(children: [
              Image.asset('assets/images/logo11.png', height: 36, fit: BoxFit.contain),
              const SizedBox(height: 16),
              Divider(color: YS.stroke),
              const SizedBox(height: 12),
              _infoRow('Company',  'YellowSense Technologies'),
              _infoRow('Project',  'UIDAI SITAA Cohort 1'),
              _infoRow('Pipeline', 'YOLO → Liveness → U²Net → ZeroDCE → MinutiaeNet'),
              _infoRow('Matching', 'MCC Graph + Relaxation Labeling'),
              _infoRow('Version',  '1.0.0'),
            ]),
          ),
          const SizedBox(height: 40),
        ]),
      ),
    );
  }

  Widget _healthCard(Map<String, dynamic> h) {
    final ok    = h['status'] == 'ok';
    final color = ok ? YS.green : YS.red;
    final bg    = ok ? YS.greenBg : YS.redBg;
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withValues(alpha: 0.3))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(ok ? Icons.check_circle_rounded : Icons.error_rounded, color: color, size: 18),
          const SizedBox(width: 8),
          Text(ok ? 'Server Online' : 'Unreachable',
              style: YS.label(14, color: color, w: FontWeight.w700)),
        ]),
        if (h['error'] != null) ...[const SizedBox(height: 8),
          Text(h['error'], style: YS.label(11, color: YS.inkMid))],
        if (ok) ...[
          const SizedBox(height: 12),
          _infoRow('Device',     h['device'] ?? '—'),
          _infoRow('Liveness',   h['liveness_available']   == true ? 'Loaded ✓' : 'Not loaded'),
          _infoRow('Minutiae',   h['minutiae_available']   == true ? 'Loaded ✓' : 'Not loaded'),
          _infoRow('Brightspot', h['brightspot_available'] == true ? 'Loaded ✓' : 'Not loaded'),
        ],
      ]),
    );
  }

  Widget _infoRow(String k, String v) => Padding(
    padding: const EdgeInsets.only(bottom: 8),
    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('$k: ', style: YS.label(12, color: YS.inkMid)),
      Expanded(child: Text(v, style: YS.label(12, w: FontWeight.w600))),
    ]),
  );
}
