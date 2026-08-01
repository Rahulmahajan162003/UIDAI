import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:permission_handler/permission_handler.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';

class FingerprintCameraWidget extends StatefulWidget {
  final void Function(File) onImageCaptured;
  final bool disabled;
  const FingerprintCameraWidget({
    super.key,
    required this.onImageCaptured,
    this.disabled = false,
  });

  @override
  State<FingerprintCameraWidget> createState() => _FingerprintCameraWidgetState();
}

class _FingerprintCameraWidgetState extends State<FingerprintCameraWidget>
    with WidgetsBindingObserver, SingleTickerProviderStateMixin {

  // ── Camera state ─────────────────────────────────────────────────────────────
  CameraController? _ctrl;
  bool _initializing = false;
  bool _live         = false;
  bool _flashing     = false;
  bool _torchOn      = false;
  File? _captured;
  String? _error;
  double _minZoom = 1.0, _maxZoom = 1.0, _zoom = 2.0;
  Offset? _focusPt;
  Timer? _focusTimer;

  // ── EC-1 fix: focus-settle delay before first poll ────────────────────────────
  static const Duration _focusSettleDelay = Duration(milliseconds: 700);

  // ── Auto-capture state ───────────────────────────────────────────────────────
  bool _autoCapture     = true;
  bool _pollInFlight    = false;   // true while takePicture+qualityCheck is running
  bool _capturing       = false;   // true once commit/fresh capture starts
  bool _pollActive      = false;   // master kill-switch for the poll loop
  String _roiGuidance   = '';
  String _guidance      = 'Place finger in the oval';
  Color  _guidanceColor = Colors.white60;
  double _qualityScore  = 0.0;
  int    _passCount     = 0;

  // EC-6 fix: consecutive-only pass counting — timestamp of last good frame
  DateTime? _lastPassTime;
  static const Duration _maxPassGap = Duration(seconds: 3);

  // EC-7 fix: max poll attempts before showing "try manually" hint
  int _pollAttempts = 0;
  static const int _maxPollAttempts = 40; // ~22 s at ~550 ms/cycle

  // Passes needed before commit
  static const int _passesNeeded = 2;

  // Cooldown between poll cycles (server time dominates anyway)
  static const Duration _pollCooldown = Duration(milliseconds: 250);

  // EC-2 fix: copy last good frame to a stable path before committing
  File? _lastGoodFrame;

  // Tips banner — hidden automatically after first quality pass
  bool _tipsVisible = true;

  // ── Scan animation ───────────────────────────────────────────────────────────
  late AnimationController _scanCtrl;
  late Animation<double>   _scanAnim;

  // ─────────────────────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _scanCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1800))
      ..repeat(reverse: true);
    _scanAnim = Tween<double>(begin: 0, end: 1)
        .animate(CurvedAnimation(parent: _scanCtrl, curve: Curves.easeInOut));
    // EC auto-launch fix: do NOT start camera here — user taps the button
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _focusTimer?.cancel();
    _scanCtrl.dispose();
    _killPollLoop();
    _ctrl?.dispose();
    super.dispose();
  }

  // EC-4/back-nav fix: stop camera & poll when app goes background or screen pops
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _killPollLoop();
      _ctrl?.dispose();
      _ctrl = null;
      if (mounted) setState(() { _live = false; _torchOn = false; });
    }
  }

  // ── Camera lifecycle ─────────────────────────────────────────────────────────

  Future<void> _startCamera() async {
    if (widget.disabled || _initializing || _live) return;
    setState(() { _initializing = true; _error = null; });
    try {
      // Check permission — it should already be granted from main() startup
      // request, but guard here in case the user denied and we need to re-ask.
      final status = await Permission.camera.status;
      if (!status.isGranted) {
        final result = await Permission.camera.request();
        if (!result.isGranted) {
          if (mounted) {
            setState(() {
              _initializing = false;
              _error = 'Camera permission denied. Enable it in Settings to continue.';
            });
          }
          return;
        }
      }

      final cams = await availableCameras();
      if (cams.isEmpty) throw Exception('No camera found');
      final cam = cams.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
          orElse: () => cams.first);

      // Dispose stale controller if any
      await _ctrl?.dispose();
      _ctrl = CameraController(cam, ResolutionPreset.high,
          enableAudio: false, imageFormatGroup: ImageFormatGroup.jpeg);
      await _ctrl!.initialize();
      await _ctrl!.setFocusMode(FocusMode.auto);
      await _ctrl!.setExposureMode(ExposureMode.auto);
      _minZoom = await _ctrl!.getMinZoomLevel();
      _maxZoom = await _ctrl!.getMaxZoomLevel();

      // EC-8 fix: only reset zoom to 2× on first launch, not on retry
      if (_zoom < _minZoom || _zoom > _maxZoom) {
        _zoom = (2.0).clamp(_minZoom, _maxZoom);
      }
      try { await _ctrl!.setZoomLevel(_zoom); } catch (_) {}
      try { await _ctrl!.setFlashMode(FlashMode.off); } catch (_) {}

      if (!mounted) return;
      setState(() { _live = true; _initializing = false; });

      // EC-1 fix: wait for autofocus to settle before starting poll
      if (_autoCapture) {
        await Future.delayed(_focusSettleDelay);
        if (mounted && _live) _kickPollLoop();
      }
    } catch (e) {
      if (mounted) setState(() { _error = e.toString(); _initializing = false; });
    }
  }

  Future<void> _stopCamera() async {
    _killPollLoop();
    if (_torchOn) {
      try { await _ctrl?.setFlashMode(FlashMode.off); } catch (_) {}
    }
    await _ctrl?.dispose();
    _ctrl = null;
    if (mounted) {
      setState(() {
        _live = false; _torchOn = false; _captured = null;
        _capturing = false; _pollInFlight = false;
        _roiGuidance = ''; _passCount = 0; _qualityScore = 0.0;
        _lastGoodFrame = null; _lastPassTime = null; _pollAttempts = 0;
        _guidance      = 'Place finger in the oval';
        _guidanceColor = Colors.white60; _focusing = false;
        _tipsVisible   = true; // reset tips for next session
      });
    }
  }

  Future<void> _toggleTorch() async {
    if (_ctrl == null) return;
    try {
      await _ctrl!.setFlashMode(_torchOn ? FlashMode.off : FlashMode.torch);
      if (mounted) setState(() => _torchOn = !_torchOn);
    } catch (_) {}
  }

  bool _focusing = false; // true for ~1.2s after a tap-to-focus

  Future<void> _onTap(TapDownDetails d, BoxConstraints c) async {
    if (_ctrl == null || !_live) return;
    final x = (d.localPosition.dx / c.maxWidth).clamp(0.0, 1.0);
    final y = (d.localPosition.dy / c.maxHeight).clamp(0.0, 1.0);

    // Show focus ring + "Focusing…" text + haptic
    if (mounted) setState(() { _focusPt = d.localPosition; _focusing = true; });
    try { HapticFeedback.lightImpact(); } catch (_) {}

    _focusTimer?.cancel();
    _focusTimer = Timer(const Duration(milliseconds: 1200), () {
      if (mounted) setState(() { _focusPt = null; _focusing = false; });
    });
    try {
      await _ctrl!.setFocusPoint(Offset(x, y));
      await _ctrl!.setExposurePoint(Offset(x, y));
    } catch (_) {}
  }

  // ── Capture helpers ──────────────────────────────────────────────────────────

  // EC-2 fix: copies the verified frame to a stable temp path so the OS
  // can't reclaim the original before enroll/auth sends it.
  Future<File?> _stableFile(File src) async {
    try {
      final dir  = Directory.systemTemp;
      final dest = File('${dir.path}/ys_fp_${DateTime.now().millisecondsSinceEpoch}.jpg');
      return await src.copy(dest.path);
    } catch (_) {
      return src; // fallback to original if copy fails
    }
  }

  // Compress a poll-only copy for /quality_check upload.
  // Target: ~150KB so upload is fast on mobile data.
  // The original full-res file is kept untouched for the final enroll/auth call.
  Future<File> _compressForPoll(File src) async {
    try {
      final dest = '${Directory.systemTemp.path}/ys_poll_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final result = await FlutterImageCompress.compressAndGetFile(
        src.absolute.path,
        dest,
        quality: 72,        // enough for blur/brightness/glare checks
        minWidth: 800,      // YOLO detection works well at 800px wide
        minHeight: 600,
        keepExif: false,
      );
      return result != null ? File(result.path) : src;
    } catch (_) {
      return src; // fallback to original if compression fails
    }
  }

  // Auto path: commits the last server-verified frame (no new photo taken)
  Future<void> _commitCapture() async {
    if (_capturing || _lastGoodFrame == null) return;
    _capturing = true;
    _killPollLoop();

    // EC-2 fix: copy to stable path
    final stable = await _stableFile(_lastGoodFrame!);
    if (stable == null) {
      if (mounted) setState(() { _capturing = false; _error = 'Capture file missing — retake'; });
      return;
    }

    if (mounted) {
      setState(() => _flashing = true);
      await Future.delayed(const Duration(milliseconds: 80));
      // EC-5 fix: check mounted again after await
      if (!mounted) return;
      setState(() { _flashing = false; _captured = stable; _live = false; });
    }
  }

  // Manual path: takes a fresh photo without quality pre-check
  Future<void> _captureFresh() async {
    if (_ctrl == null || !_ctrl!.value.isInitialized || _capturing) return;
    _capturing = true;
    _killPollLoop();
    if (mounted) setState(() => _flashing = true);
    await Future.delayed(const Duration(milliseconds: 80));
    if (!mounted) return;
    setState(() => _flashing = false);
    try {
      final xf   = await _ctrl!.takePicture();
      final stable = await _stableFile(File(xf.path));
      if (!mounted) return;
      setState(() { _captured = stable; _live = false; });
    } catch (e) {
      if (mounted) {
        setState(() { _error = 'Capture failed: $e'; _capturing = false; });
        if (_autoCapture) _kickPollLoop();
      }
    }
  }

  void _useImage() {
    if (_captured != null) widget.onImageCaptured(_captured!);
  }

  Future<void> _retry() async {
    // EC-8 fix: preserve user's current zoom on retry
    setState(() {
      _captured = null; _live = false; _capturing = false;
      _pollInFlight = false; _roiGuidance = ''; _passCount = 0;
      _qualityScore = 0.0; _lastGoodFrame = null; _lastPassTime = null;
      _pollAttempts = 0; _error = null; _tipsVisible = true;
      _guidance      = 'Place finger in the oval';
      _guidanceColor = Colors.white60;
    });
    await _startCamera();
  }

  // ── Poll loop ────────────────────────────────────────────────────────────────
  //
  // Design:
  //   • Fire-on-completion chain — only ONE request ever in-flight (EC-3 fixed)
  //   • Focus settle delay before first poll (EC-1)
  //   • Consecutive-pass validation — gap > 3s resets count (EC-6)
  //   • Max 40 attempts then surfaced as hint (EC-7)
  //   • Kill-switch (_pollActive) checked at every async boundary (EC-4/5)

  void _kickPollLoop() {
    if (_pollActive) return; // already running
    _pollActive   = true;
    _pollInFlight = false;
    _passCount    = 0;
    _lastGoodFrame = null;
    _lastPassTime  = null;
    _pollAttempts  = 0;
    _roiGuidance   = '';
    Future.microtask(_pollOnce);
  }

  void _killPollLoop() {
    _pollActive   = false;
    _pollInFlight = false;
  }

  Future<void> _pollOnce() async {
    // All guard checks at the top
    if (!_pollActive || !_live || _capturing || _captured != null) return;
    if (_ctrl == null || !_ctrl!.value.isInitialized) return;
    if (_pollInFlight) return; // EC-3: hard re-entry guard

    // EC-7: max attempts guard
    if (_pollAttempts >= _maxPollAttempts) {
      if (mounted) {
        setState(() {
          _guidance      = 'Switch to MANUAL — auto not detecting';
          _guidanceColor = Colors.white38;
        });
      }
      _pollActive = false;
      return;
    }

    _pollInFlight = true;
    _pollAttempts++;

    try {
      // ── Step 1: capture poll frame ──────────────────────────────────────────
      final xf   = await _ctrl!.takePicture();
      final file = File(xf.path); // original full-res — kept for final commit

      if (!_pollActive || _capturing) return; // killed mid-flight

      // Compress a small copy for the quality check upload.
      // Original stays untouched — will be used as the final capture image.
      final pollFile = await _compressForPoll(file);

      // ── Step 2: quality check (6s timeout) ─────────────────────────────────
      final result = await ApiService.qualityCheck(pollFile)
          .timeout(const Duration(seconds: 6), onTimeout: () => {});

      // EC-5: check mounted + active after every await
      if (!mounted || !_pollActive || _capturing) return;

      if (result.isEmpty) {
        // Network timeout — skip, don't penalise pass count or attempts
        _pollAttempts = (_pollAttempts - 1).clamp(0, _maxPollAttempts);
        _scheduleNextPoll();
        return;
      }

      final passed  = result['passed'] == true;
      final guidance = result['guidance'] as String? ?? 'Checking...';

      // Quality score for the progress bar
      final blurOk   = result['blur']?['is_blurry']        != true ? 1.0 : 0.0;
      final brightOk = (result['brightness']?['too_dark']   == true ||
                        result['brightness']?['too_bright']  == true) ? 0.0 : 1.0;
      final glareOk  = result['glare']?['has_glare']        != true ? 1.0 : 0.0;
      final score    = (blurOk * 0.4 + brightOk * 0.35 + glareOk * 0.25)
          .clamp(0.0, 1.0);

      final inRoi    = result['in_roi']       as bool?   ?? true;
      final roiGuide = result['roi_guidance'] as String? ?? '';

      if (mounted) {
        setState(() {
          _guidance      = passed ? '✓  Good — hold still' : guidance;
          _guidanceColor = passed ? YS.amber : Colors.orangeAccent;
          _qualityScore  = score;
          _roiGuidance   = (!inRoi && roiGuide.isNotEmpty) ? roiGuide : '';
        });
      }

      if (passed) {
        // EC-6: reset pass count if gap since last good frame is too large
        final now = DateTime.now();
        if (_lastPassTime != null &&
            now.difference(_lastPassTime!) > _maxPassGap) {
          _passCount = 0;
        }
        _lastPassTime  = now;
        _lastGoodFrame = file;
        _passCount++;
        // Collapse tips banner once quality passes — user has understood the setup
        if (_tipsVisible && mounted) setState(() => _tipsVisible = false);

        if (_passCount >= _passesNeeded && _pollActive && !_capturing) {
          _pollInFlight = false;
          await _commitCapture();
          return;
        }
      } else {
        // Failed frame — reset streak
        _passCount     = 0;
        _lastGoodFrame = null;
        _lastPassTime  = null;
      }
    } catch (_) {
      // Swallow — could be camera disposed mid-flight
      if (mounted && _pollActive) {
        setState(() {
          _guidance      = 'Auto-check unavailable — tap MANUAL';
          _guidanceColor = Colors.white38;
        });
      }
    } finally {
      _pollInFlight = false;
    }

    _scheduleNextPoll();
  }

  void _scheduleNextPoll() {
    if (!_pollActive) return;
    Future.delayed(_pollCooldown, () {
      if (_pollActive && !_capturing && _captured == null) _pollOnce();
    });
  }

  void _toggleAutoCapture() {
    final next = !_autoCapture;
    setState(() => _autoCapture = next);
    if (_live) {
      if (next) {
        _kickPollLoop();
      } else {
        _killPollLoop();
        setState(() {
          _guidance      = 'Place finger in the oval';
          _guidanceColor = Colors.white60;
          _qualityScore  = 0.0;
          _passCount     = 0;
          _lastGoodFrame = null;
          _roiGuidance   = '';
        });
      }
    }
  }

  /// Maps ROI guidance text to a directional icon
  IconData _roiGuidanceIcon(String guidance) {
    final g = guidance.toLowerCase();
    if (g.contains('left'))  return Icons.arrow_back_rounded;
    if (g.contains('right')) return Icons.arrow_forward_rounded;
    if (g.contains('up') || g.contains('higher'))   return Icons.arrow_upward_rounded;
    if (g.contains('down') || g.contains('lower'))  return Icons.arrow_downward_rounded;
    if (g.contains('closer') || g.contains('near')) return Icons.zoom_in_rounded;
    if (g.contains('further') || g.contains('far')) return Icons.zoom_out_rounded;
    return Icons.open_with_rounded;
  }

  // ── Build ────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: YS.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: YS.stroke),
      ),
      clipBehavior: Clip.hardEdge,
      child: Column(children: [_header(), _viewfinder(), _controls()]),
    );
  }

  Widget _header() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
    decoration: const BoxDecoration(
        color: YS.card,
        border: Border(bottom: BorderSide(color: YS.stroke))),
    child: Row(children: [
      Text('FINGERPRINT',
          style: YS.label(11, color: YS.amber, w: FontWeight.w800)
              .copyWith(letterSpacing: 1.5)),
      const Spacer(),
      // AUTO / MANUAL toggle — only shown when camera is live
      if (_live)
        GestureDetector(
          onTap: _toggleAutoCapture,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(8),
              color: _autoCapture ? YS.amberSoft : YS.cardAlt,
              border: Border.all(color: _autoCapture ? YS.amber : YS.stroke),
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(
                _autoCapture
                    ? Icons.auto_mode_rounded
                    : Icons.touch_app_rounded,
                size: 12,
                color: _autoCapture ? YS.amberDeep : YS.inkLight,
              ),
              const SizedBox(width: 4),
              Text(_autoCapture ? 'AUTO' : 'MANUAL',
                  style: YS.label(9,
                      color: _autoCapture ? YS.amberDeep : YS.inkLight,
                      w: FontWeight.w700)),
            ]),
          ),
        ),
      if (_live) const SizedBox(width: 8),
      // Torch toggle — only when camera is live
      if (_live)
        GestureDetector(
          onTap: _toggleTorch,
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: _torchOn ? YS.amber : YS.stroke),
              color: _torchOn ? YS.amberSoft : YS.cardAlt,
            ),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Icon(_torchOn ? Icons.flash_on : Icons.flash_off,
                  size: 12,
                  color: _torchOn ? YS.amberDeep : YS.inkLight),
              const SizedBox(width: 4),
              Text(_torchOn ? 'ON' : 'OFF',
                  style: YS.label(9,
                      color: _torchOn ? YS.amberDeep : YS.inkLight,
                      w: FontWeight.w700)),
            ]),
          ),
        ),
    ]),
  );

  Widget _viewfinder() => LayoutBuilder(
    builder: (context, constraints) {
      final viewW = constraints.maxWidth;
      final viewH = viewW * 4 / 3;
      return SizedBox(
        width: viewW, height: viewH,
        child: Container(
          color: const Color(0xFF1A1A1A),
          child: Stack(fit: StackFit.expand, children: [

            // ── Live camera preview ───────────────────────────────────────────
            if (_live && _ctrl != null && _ctrl!.value.isInitialized)
              GestureDetector(
                onTapDown: (d) => _onTap(d, constraints),
                child: CameraPreview(_ctrl!),
              ),

            // ── Captured image preview ────────────────────────────────────────
            if (_captured != null)
              Image.file(_captured!, fit: BoxFit.cover),

            // ── Idle state — camera not started ──────────────────────────────
            if (!_live && !_initializing && _captured == null)
              Container(
                color: const Color(0xFF1A1A1A),
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 72, height: 72,
                        decoration: BoxDecoration(
                          color: YS.amber.withValues(alpha: 0.12),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(Icons.fingerprint_rounded,
                            size: 40, color: YS.amber),
                      ),
                      const SizedBox(height: 14),
                      Text('Tap "Start Camera" below',
                          style: YS.label(12, color: Colors.white38,
                              w: FontWeight.w500)),
                    ],
                  ),
                ),
              ),

            // ── Initializing spinner ──────────────────────────────────────────
            if (_initializing)
              Container(
                color: const Color(0xFF1A1A1A),
                child: const Center(child: CircularProgressIndicator(
                    color: YS.amber, strokeWidth: 2)),
              ),

            // ── Oval overlay (always on top of preview) ───────────────────────
            if (_live || _captured != null)
              CustomPaint(painter: _OverlayPainter(YS.amber)),

            // ── Scan line (auto mode only) ────────────────────────────────────
            if (_live && _autoCapture)
              AnimatedBuilder(
                animation: _scanAnim,
                builder: (_, __) => Positioned(
                  top: _scanAnim.value * (viewH * 0.6),
                  left: viewW * 0.18,
                  right: viewW * 0.18,
                  child: Container(
                    height: 2,
                    decoration: const BoxDecoration(
                      gradient: LinearGradient(colors: [
                        Colors.transparent, YS.amber, Colors.transparent,
                      ]),
                    ),
                  ),
                ),
              ),

            // ── Focus ring ────────────────────────────────────────────────────
            if (_focusPt != null)
              Positioned(
                left: _focusPt!.dx - 25, top: _focusPt!.dy - 25,
                child: Container(
                  width: 50, height: 50,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    border: Border.all(color: YS.amber, width: 2),
                  ),
                ),
              ),

            // ── Flash overlay ─────────────────────────────────────────────────
            if (_flashing)
              Container(color: Colors.white.withValues(alpha: 0.8)),

            // ── Quality bar (auto mode, live only) ────────────────────────────
            if (_live && _autoCapture)
              Positioned(
                top: 12, left: 16, right: 16,
                child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                    Text('QUALITY',
                        style: YS.label(9,
                            color: Colors.white54, w: FontWeight.w700)
                            .copyWith(letterSpacing: 1.5)),
                    // Pass dots
                    Row(children: List.generate(_passesNeeded, (i) =>
                      Container(
                        width: 7, height: 7,
                        margin: const EdgeInsets.only(left: 4),
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: i < _passCount
                              ? YS.amber
                              : Colors.white.withValues(alpha: 0.25),
                        ),
                      ))),
                  ]),
                  const SizedBox(height: 4),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: _qualityScore,
                      minHeight: 4,
                      backgroundColor: Colors.white.withValues(alpha: 0.15),
                      color: _qualityScore > 0.7
                          ? YS.amber
                          : _qualityScore > 0.4
                              ? Colors.orangeAccent
                              : Colors.redAccent,
                    ),
                  ),
                ]),
              ),

            // ── Guidance + ROI text ───────────────────────────────────────────
            if (_live || _captured != null)
              Positioned(
                bottom: 14, left: 16, right: 16,
                child: Column(mainAxisSize: MainAxisSize.min, children: [

                  // ROI directional guidance — larger, clearer, icon + text
                  if (_roiGuidance.isNotEmpty && _live && _autoCapture)
                    Container(
                      margin: const EdgeInsets.only(bottom: 6),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 14, vertical: 7),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.75),
                        borderRadius: BorderRadius.circular(22),
                        border: Border.all(
                            color: Colors.orangeAccent.withValues(alpha: 0.5)),
                      ),
                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                        Icon(
                          _roiGuidanceIcon(_roiGuidance),
                          size: 15, color: Colors.orangeAccent),
                        const SizedBox(width: 6),
                        Text(_roiGuidance,
                            style: YS.label(12,
                                color: Colors.orangeAccent,
                                w: FontWeight.w700)),
                      ]),
                    ),

                  // Focusing feedback (shown for 1.2s after tap)
                  if (_focusing && _live)
                    Container(
                      margin: const EdgeInsets.only(bottom: 6),
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 5),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                        const SizedBox(width: 10, height: 10,
                            child: CircularProgressIndicator(
                                strokeWidth: 1.5,
                                color: YS.amber)),
                        const SizedBox(width: 6),
                        Text('Focusing…',
                            style: YS.label(11,
                                color: YS.amber, w: FontWeight.w600)),
                      ]),
                    ),

                  // Main quality/guidance text
                  if (_live)
                    Text(
                      _guidance,
                      textAlign: TextAlign.center,
                      style: YS.label(11,
                          color: _guidanceColor, w: FontWeight.w500),
                    ),
                  if (_captured != null)
                    Text('✓  Image captured',
                        textAlign: TextAlign.center,
                        style: YS.label(11,
                            color: YS.amber, w: FontWeight.w500)),
                ]),
              ),
          ]),
        ),
      );
    },
  );

  Widget _controls() => Container(
    color: YS.card,
    padding: const EdgeInsets.all(16),
    child: Column(children: [

      // Zoom slider — only when live and device supports zoom
      if (_maxZoom > _minZoom + 0.1 && _live)
        Row(children: [
          const Icon(Icons.zoom_out, size: 14, color: YS.inkLight),
          Expanded(child: Slider(
            value: _zoom.clamp(_minZoom, _maxZoom.clamp(_minZoom, 5.0)),
            min: _minZoom,
            max: _maxZoom.clamp(_minZoom, 5.0),
            activeColor: YS.amber,
            inactiveColor: YS.stroke,
            thumbColor: YS.amber,
            onChanged: (v) async {
              // EC-8: update _zoom so retries respect the user's choice
              setState(() => _zoom = v);
              try { await _ctrl?.setZoomLevel(v); } catch (_) {}
            },
          )),
          const Icon(Icons.zoom_in, size: 14, color: YS.inkLight),
          const SizedBox(width: 4),
          Text('${_zoom.toStringAsFixed(1)}×',
              style: YS.label(10, color: YS.inkLight)),
        ]),

      // ── Tips banner — shown until first quality pass or manually dismissed ──
      if (_live && _captured == null && _tipsVisible) ...[
        const SizedBox(height: 10),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          decoration: BoxDecoration(
            color: YS.cardAlt,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: YS.stroke),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(children: [
                Text('Tips for best capture',
                    style: YS.label(11, color: YS.inkMid, w: FontWeight.w700)
                        .copyWith(letterSpacing: 0.3)),
                const Spacer(),
                GestureDetector(
                  onTap: () => setState(() => _tipsVisible = false),
                  child: const Icon(Icons.close_rounded, size: 14, color: YS.inkLight),
                ),
              ]),
              const SizedBox(height: 6),
              _tip(Icons.touch_app_rounded,
                  'Tap anywhere on the preview to refocus'),
              _tip(Icons.flash_on_rounded,
                  'Dark room? Tap the torch button (⚡) above to turn on flash'),
              _tip(Icons.zoom_in_rounded,
                  'Use the zoom slider to fill the oval with your finger — 2–3× works best'),
              _tip(Icons.open_with_rounded,
                  'Follow the on-screen arrows to centre your finger in the oval'),
            ],
          ),
        ),
      ],

      // ── Idle: Start Camera button ─────────────────────────────────────────
      if (!_live && !_initializing && _captured == null)
        _btn('Start Camera', _startCamera, true,
            icon: Icons.camera_alt_rounded),

      // ── Initializing ─────────────────────────────────────────────────────
      if (_initializing)
        Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            color: YS.amberSoft, borderRadius: BorderRadius.circular(12)),
          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            const SizedBox(width: 14, height: 14,
                child: CircularProgressIndicator(
                    strokeWidth: 2, color: YS.amberDeep)),
            const SizedBox(width: 10),
            Text('Starting camera…',
                style: YS.label(13, color: YS.amberDeep, w: FontWeight.w600)),
          ]),
        ),

      // ── Live + Manual mode ────────────────────────────────────────────────
      if (_live && !_autoCapture)
        Row(children: [
          Expanded(child: _btn('Capture', _captureFresh, true,
              icon: Icons.camera_rounded)),
          const SizedBox(width: 10),
          _btn('✕', _stopCamera, false, square: true),
        ]),

      // ── Live + Auto mode — scanning status ────────────────────────────────
      if (_live && _autoCapture)
        Row(children: [
          Expanded(
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 14),
              decoration: BoxDecoration(
                color: YS.amberSoft,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: YS.amber.withValues(alpha: 0.4)),
              ),
              child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                SizedBox(
                  width: 14, height: 14,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: YS.amberDeep,
                    value: _passCount > 0
                        ? _passCount / _passesNeeded
                        : null,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  _passCount > 0
                      ? 'Hold still… $_passCount/$_passesNeeded'
                      : 'Scanning quality…',
                  style: YS.label(13,
                      color: YS.amberDeep, w: FontWeight.w600),
                ),
              ]),
            ),
          ),
          const SizedBox(width: 10),
          _btn('✕', _stopCamera, false, square: true),
        ]),

      // ── Post-capture ──────────────────────────────────────────────────────
      if (_captured != null)
        Row(children: [
          Expanded(child: _btn('Use This Image', _useImage, true,
              icon: Icons.check_rounded)),
          const SizedBox(width: 10),
          Expanded(child: _btn('Retake', _retry, false,
              icon: Icons.refresh_rounded)),
        ]),

      // Error message
      if (_error != null) ...[
        const SizedBox(height: 8),
        Text(_error!,
            style: const TextStyle(color: YS.red, fontSize: 11)),
      ],
    ]),
  );

  Widget _tip(IconData icon, String text) => Padding(
    padding: const EdgeInsets.only(top: 4),
    child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Icon(icon, size: 12, color: YS.amber),
      const SizedBox(width: 6),
      Expanded(child: Text(text,
          style: YS.label(11, color: YS.inkLight))),
    ]),
  );

  Widget _btn(String label, VoidCallback onTap, bool primary,
      {bool square = false, IconData? icon}) =>
      GestureDetector(
        onTap: widget.disabled ? null : onTap,
        child: Container(
          width: square ? 48 : null,
          padding: EdgeInsets.symmetric(
              horizontal: square ? 0 : 14, vertical: 13),
          decoration: BoxDecoration(
            color: primary ? YS.amber : YS.cardAlt,
            borderRadius: BorderRadius.circular(12),
            border: primary ? null : Border.all(color: YS.stroke),
          ),
          child: square
              ? Center(child: Text(label,
                  textAlign: TextAlign.center,
                  style: YS.label(13,
                      color: primary ? Colors.black : YS.inkMid,
                      w: FontWeight.w700)))
              : Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                  if (icon != null) ...[
                    Icon(icon,
                        size: 14,
                        color: primary ? Colors.black : YS.inkMid),
                    const SizedBox(width: 6),
                  ],
                  Text(label,
                      style: YS.label(13,
                          color: primary ? Colors.black : YS.inkMid,
                          w: FontWeight.w700)),
                ]),
        ),
      );
}

// ── Oval overlay painter ──────────────────────────────────────────────────────

class _OverlayPainter extends CustomPainter {
  final Color color;
  const _OverlayPainter(this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2, cy = size.height / 2;
    final rx = size.width * 0.23, ry = size.height * 0.25;
    final oval = Rect.fromCenter(
        center: Offset(cx, cy), width: rx * 2, height: ry * 2);
    canvas.drawPath(
      Path()
        ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
        ..addOval(oval)
        ..fillType = PathFillType.evenOdd,
      Paint()..color = const Color(0xFF050505).withValues(alpha: 0.65),
    );
    canvas.drawOval(
      oval,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3),
    );
    canvas.drawOval(
      oval,
      Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.2,
    );
  }

  @override
  bool shouldRepaint(_OverlayPainter old) => old.color != color;
}
