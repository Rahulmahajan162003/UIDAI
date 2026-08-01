import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../theme/app_theme.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: YS.bg,
      body: SafeArea(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            // ── Top bar ─────────────────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
              child: Row(children: [
                Image.asset('assets/images/logo11.png', height: 28, fit: BoxFit.contain),
                const Spacer(),
                GestureDetector(
                  onTap: () => context.go('/settings'),
                  child: Container(
                    width: 38, height: 38,
                    decoration: BoxDecoration(
                      color: YS.card, borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: YS.stroke),
                      boxShadow: YS.cardShadow,
                    ),
                    child: const Icon(Icons.settings_outlined, size: 18, color: YS.inkMid),
                  ),
                ),
              ]),
            ),

            // ── Hero banner ─────────────────────────────────────────────────
            Container(
              margin: const EdgeInsets.fromLTRB(20, 20, 20, 0),
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  begin: Alignment.topLeft, end: Alignment.bottomRight,
                  colors: [Color(0xFFE0A020), Color(0xFFB87800)],
                ),
                borderRadius: BorderRadius.circular(24),
                boxShadow: YS.amberShadow,
              ),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text('UIDAI · SITAA Cohort 1',
                      style: YS.label(11, color: Colors.white, w: FontWeight.w600)),
                ),
                const SizedBox(height: 14),
                Text('Contactless\nFingerprint Auth',
                    style: YS.display(28, color: Colors.white, w: FontWeight.w800)),
                const SizedBox(height: 8),
                Text('AI-powered biometric authentication\nfor Aadhaar — no scanner needed',
                    style: YS.label(13, color: Colors.white.withValues(alpha: 0.85))),
                const SizedBox(height: 20),
                // Pipeline chips
                Wrap(spacing: 8, runSpacing: 8, children: [
                  _chip('YOLO'), _chip('U²Net'), _chip('ZeroDCE'),
                  _chip('MinutiaeNet'), _chip('MCC'),
                ]),
              ]),
            ),

            // ── Stats row ───────────────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
              child: Row(children: [
                _statCard('5', 'AI Models', Icons.memory_rounded),
                const SizedBox(width: 10),
                _statCard('1:N', 'Auth', Icons.groups_rounded),
                const SizedBox(width: 10),
                _statCard('1:1', 'Verify', Icons.person_search_rounded),
              ]),
            ),

            // ── Section label ───────────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 28, 20, 12),
              child: Text('ACTIONS',
                  style: YS.label(11, color: YS.inkLight, w: FontWeight.w700)
                      .copyWith(letterSpacing: 1.8)),
            ),

            // ── Action cards ────────────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(children: [
                // Primary actions — 2 column
                Row(children: [
                  Expanded(child: _BigCard(
                    icon: Icons.fingerprint_rounded,
                    label: 'Enroll',
                    sub: 'Register user',
                    color: YS.amber,
                    bg: YS.amberSoft,
                    onTap: () => context.go('/enroll'),
                  )),
                  const SizedBox(width: 12),
                  Expanded(child: _BigCard(
                    icon: Icons.how_to_reg_rounded,
                    label: 'Authenticate',
                    sub: '1:N matching',
                    color: YS.green,
                    bg: YS.greenBg,
                    onTap: () => context.go('/authenticate'),
                  )),
                ]),
                const SizedBox(height: 12),
                Row(children: [
                  Expanded(child: _BigCard(
                    icon: Icons.verified_rounded,
                    label: 'Verify',
                    sub: '1:1 identity',
                    color: YS.blue,
                    bg: YS.blueBg,
                    onTap: () => context.go('/verify'),
                  )),
                  const SizedBox(width: 12),
                  Expanded(child: _BigCard(
                    icon: Icons.biotech_rounded,
                    label: 'Pipeline',
                    sub: 'Live visualizer',
                    color: YS.orange,
                    bg: YS.orangeBg,
                    onTap: () => context.go('/pipeline'),
                  )),
                ]),
                const SizedBox(height: 12),
                // Full width QC pipeline tool
                _WideCard(
                  icon: Icons.science_rounded,
                  label: 'QC Pipeline',
                  sub: 'Quality check & readiness scoring',
                  color: YS.orange,
                  onTap: () => context.go('/qc'),
                ),
                const SizedBox(height: 12),
                // Full width history
                _WideCard(
                  icon: Icons.history_rounded,
                  label: 'Auth History',
                  sub: 'View authentication records and logs',
                  color: YS.inkMid,
                  onTap: () => context.go('/history'),
                ),
              ]),
            ),

            // ── Footer ──────────────────────────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 28, 20, 24),
              child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                Text('Powered by ', style: YS.label(11, color: YS.inkLight)),
                Text('YellowSense Technologies',
                    style: YS.label(11, color: YS.amberDeep, w: FontWeight.w700)),
                Text(' · v1.0', style: YS.label(11, color: YS.inkLight)),
              ]),
            ),
          ],
        ),
      ),
    );
  }

  Widget _chip(String label) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
    decoration: BoxDecoration(
      color: Colors.white.withValues(alpha: 0.2),
      borderRadius: BorderRadius.circular(20),
      border: Border.all(color: Colors.white.withValues(alpha: 0.3)),
    ),
    child: Text(label, style: YS.label(10, color: Colors.white, w: FontWeight.w600)),
  );

  Widget _statCard(String val, String label, IconData icon) => Expanded(
    child: Container(
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 12),
      decoration: BoxDecoration(
        color: YS.card, borderRadius: BorderRadius.circular(14),
        border: Border.all(color: YS.stroke),
        boxShadow: YS.cardShadow,
      ),
      child: Column(children: [
        Icon(icon, size: 18, color: YS.amber),
        const SizedBox(height: 6),
        Text(val, style: YS.display(18, color: YS.ink)),
        const SizedBox(height: 2),
        Text(label, style: YS.label(10, color: YS.inkLight)),
      ]),
    ),
  );
}

class _BigCard extends StatelessWidget {
  final IconData icon;
  final String label, sub;
  final Color color, bg;
  final VoidCallback onTap;
  const _BigCard({required this.icon, required this.label, required this.sub,
      required this.color, required this.bg, required this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onTap,
    child: Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: YS.card, borderRadius: BorderRadius.circular(18),
        border: Border.all(color: YS.stroke),
        boxShadow: YS.cardShadow,
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Container(
          width: 44, height: 44,
          decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(12)),
          child: Icon(icon, color: color, size: 22),
        ),
        const SizedBox(height: 14),
        Text(label, style: YS.label(15, color: YS.ink, w: FontWeight.w700)),
        const SizedBox(height: 2),
        Text(sub, style: YS.label(12, color: YS.inkLight)),
      ]),
    ),
  );
}

class _WideCard extends StatelessWidget {
  final IconData icon;
  final String label, sub;
  final Color color;
  final VoidCallback onTap;
  const _WideCard({required this.icon, required this.label, required this.sub,
      required this.color, required this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
    onTap: onTap,
    child: Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: YS.card, borderRadius: BorderRadius.circular(18),
        border: Border.all(color: YS.stroke),
        boxShadow: YS.cardShadow,
      ),
      child: Row(children: [
        Container(
          width: 44, height: 44,
          decoration: BoxDecoration(color: YS.cardAlt, borderRadius: BorderRadius.circular(12)),
          child: Icon(icon, color: color, size: 22),
        ),
        const SizedBox(width: 14),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(label, style: YS.label(15, color: YS.ink, w: FontWeight.w700)),
          Text(sub, style: YS.label(12, color: YS.inkLight)),
        ])),
        const Icon(Icons.chevron_right_rounded, color: YS.inkFaint, size: 20),
      ]),
    ),
  );
}
