"""
identity_store.py — person-centric enrolment and 1:1 verification, keyed by a
caller-supplied 12-character alphanumeric UID.

This is the shared layer behind the /uid/* endpoints. Both Flask servers
(app.py :5002 single finger, slap_app.py :5010 slap) import it, so one person has
ONE UID whichever capture mode enrolled them. It owns its own SQLite file
(identity.db) and never touches uidai.db or slap.db — the older endpoints are
unaffected.

Rules:
    * UID is exactly 12 letters/digits, uppercased on input.
    * Single-finger probes match only single-finger samples; slap probes match
      only slap samples (the `modality` column).
    * A probe finger is scored only against samples at the SAME finger_position,
      best score over that finger's samples (up to MAX_SAMPLES of them).
    * Slap verify passes when at least MIN_FINGERS_PASS fingers pass; a UID with
      fewer than MIN_FINGERS_PASS enrolled slap fingers is rejected outright.

Config is read from the environment at call time:
    API_KEY               required by the /uid/* endpoints; they refuse without it
    UID_MATCH_THRESH      per-finger pass threshold (default 0.40, UNCALIBRATED)
    UID_MIN_FINGERS_PASS  slap fingers that must pass (default 2)
    UID_MAX_SAMPLES       samples kept per finger (default 5)
    UID_MATCH_ALGORITHM   matching.py algorithm name (default "ransac_alignment")

Why ransac_alignment and 0.40: on synthetic rotated, partially overlapping
recaptures (see tests/test_matching.py recapture()) it separated best — genuine
minimum 0.60 vs impostor maximum 0.23. "modern" scores random impostors ~0.38 on
average, so a 0.25 threshold is meaningless for it. Both values are a starting
point from SYNTHETIC data only; recalibrate from a measured FAR/FRR curve.
    IDENTITY_DB           path to the SQLite file (default backend/identity.db)

No TensorFlow or model weights are needed to import this module.
"""

import hmac
import json
import os
import re
import sqlite3
from datetime import datetime
from functools import wraps

from flask import Blueprint, jsonify, request

from matching import match_templates

MODALITIES = ("single", "slap")
UID_PATTERN = re.compile(r"^[A-Z0-9]{12}$")
MIN_MINUTIAE = 5


# ── configuration ────────────────────────────────────────────────────────────
def db_path():
    return os.environ.get(
        "IDENTITY_DB",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "identity.db"),
    )


def match_threshold():
    return float(os.environ.get("UID_MATCH_THRESH", 0.40))


def min_fingers_pass():
    return int(os.environ.get("UID_MIN_FINGERS_PASS", 2))


def max_samples():
    return int(os.environ.get("UID_MAX_SAMPLES", 5))


def match_algorithm():
    return os.environ.get("UID_MATCH_ALGORITHM", "ransac_alignment")


# ── errors ───────────────────────────────────────────────────────────────────
class IdentityError(Exception):
    """Base for errors that map to a clean HTTP response."""
    status = 400
    code = "identity_error"

    def __init__(self, message=None, **extra):
        super().__init__(message or self.code)
        self.extra = extra


class InvalidUid(IdentityError):
    status, code = 400, "invalid_uid"


class UidExists(IdentityError):
    status, code = 409, "uid_exists"


class PersonNotFound(IdentityError):
    status, code = 404, "uid_not_found"


class SampleCapExceeded(IdentityError):
    status, code = 409, "sample_cap_reached"


class FingerNotEnrolled(IdentityError):
    status, code = 409, "finger_not_enrolled"


class NotEnoughEnrolledFingers(IdentityError):
    status, code = 409, "not_enough_enrolled_fingers"


class BadRequest(IdentityError):
    status, code = 400, "bad_request"


# ── validation ───────────────────────────────────────────────────────────────
def normalize_uid(raw):
    uid = (raw or "").strip().upper()
    if not UID_PATTERN.match(uid):
        raise InvalidUid("uid must be exactly 12 letters/digits")
    return uid


def normalize_position(modality, raw):
    """Single-finger positions are ISO 19794 codes 1-10; slap positions are the
    names slap_app.py assigns (e.g. 'right_index'). Stored as text either way."""
    pos = str(raw if raw is not None else "").strip().lower()
    if modality == "single":
        if not pos.isdigit() or not 1 <= int(pos) <= 10:
            raise BadRequest("finger_position must be an integer 1-10")
        return str(int(pos))
    if not pos:
        raise BadRequest("finger_position is required")
    return pos


# ── database ─────────────────────────────────────────────────────────────────
def _connect():
    con = sqlite3.connect(db_path(), timeout=10)
    con.row_factory = sqlite3.Row
    return con


def init_identity_db():
    con = _connect()
    try:
        con.execute("PRAGMA journal_mode=WAL")  # two servers share this file
        con.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                uid         TEXT PRIMARY KEY,
                name        TEXT,
                created_at  TEXT NOT NULL
            )""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS person_templates (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                uid             TEXT NOT NULL REFERENCES persons(uid),
                modality        TEXT NOT NULL,
                finger_position TEXT NOT NULL,
                minutiae_json   TEXT NOT NULL,
                minutiae_count  INTEGER NOT NULL,
                created_at      TEXT NOT NULL
            )""")
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_person_templates_lookup
                ON person_templates(uid, modality, finger_position)""")
        con.execute("""
            CREATE TABLE IF NOT EXISTS person_verifications (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                uid         TEXT NOT NULL,
                modality    TEXT NOT NULL,
                matched     INTEGER NOT NULL,
                score       REAL,
                created_at  TEXT NOT NULL
            )""")
        con.commit()
    finally:
        con.close()


def _now():
    return datetime.utcnow().isoformat()


# ── persons ──────────────────────────────────────────────────────────────────
def register_person(uid, name=None):
    uid = normalize_uid(uid)
    name = (name or "").strip() or None
    con = _connect()
    try:
        try:
            con.execute("INSERT INTO persons (uid,name,created_at) VALUES (?,?,?)",
                        (uid, name, _now()))
            con.commit()
        except sqlite3.IntegrityError:
            raise UidExists("uid already registered")
        row = con.execute("SELECT * FROM persons WHERE uid=?", (uid,)).fetchone()
        return dict(row)
    finally:
        con.close()


def _require_person(con, uid):
    row = con.execute("SELECT * FROM persons WHERE uid=?", (uid,)).fetchone()
    if row is None:
        raise PersonNotFound("uid not registered")
    return row


def enrolled_summary(uid):
    uid = normalize_uid(uid)
    con = _connect()
    try:
        person = _require_person(con, uid)
        enrolled = {m: {} for m in MODALITIES}
        rows = con.execute(
            "SELECT modality, finger_position, COUNT(*) AS n FROM person_templates "
            "WHERE uid=? GROUP BY modality, finger_position", (uid,)).fetchall()
        for r in rows:
            enrolled[r["modality"]][r["finger_position"]] = r["n"]
        return {"uid": uid, "name": person["name"], "created_at": person["created_at"],
                "enrolled": enrolled, "max_samples_per_finger": max_samples()}
    finally:
        con.close()


def rename_person(uid, name):
    uid = normalize_uid(uid)
    name = (name or "").strip()
    if not name:
        raise BadRequest("name is required")
    con = _connect()
    try:
        _require_person(con, uid)
        con.execute("UPDATE persons SET name=? WHERE uid=?", (name, uid))
        con.commit()
    finally:
        con.close()
    return enrolled_summary(uid)


def delete_finger(uid, modality, finger_position):
    """Remove every sample for one finger. Returns how many were deleted."""
    if modality not in MODALITIES:
        raise BadRequest("modality must be 'single' or 'slap'")
    uid = normalize_uid(uid)
    pos = normalize_position(modality, finger_position)
    con = _connect()
    try:
        _require_person(con, uid)
        n = con.execute(
            "DELETE FROM person_templates WHERE uid=? AND modality=? AND finger_position=?",
            (uid, modality, pos)).rowcount
        if n == 0:
            raise FingerNotEnrolled("no samples for this finger", finger_position=pos)
        con.commit()
        return n
    finally:
        con.close()


def delete_person(uid):
    """Hard delete: the person, all their templates and their verification log."""
    uid = normalize_uid(uid)
    con = _connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        _require_person(con, uid)
        templates = con.execute("DELETE FROM person_templates WHERE uid=?", (uid,)).rowcount
        con.execute("DELETE FROM person_verifications WHERE uid=?", (uid,))
        con.execute("DELETE FROM persons WHERE uid=?", (uid,))
        con.commit()
        return templates
    except Exception:
        if con.in_transaction:
            con.rollback()
        raise
    finally:
        con.close()


# ── enrolment ────────────────────────────────────────────────────────────────
def add_samples(uid, modality, samples):
    """Add one sample per (finger_position, minutiae) pair, atomically.

    If any finger is already at the cap, nothing is written. Returns
    {finger_position: samples_now_stored}."""
    if modality not in MODALITIES:
        raise BadRequest("unknown modality")
    uid = normalize_uid(uid)
    cap = max_samples()
    samples = [(normalize_position(modality, pos), m) for pos, m in samples]
    if not samples:
        raise BadRequest("no samples to store")

    con = _connect()
    try:
        con.execute("BEGIN IMMEDIATE")  # cap check and insert must not interleave
        _require_person(con, uid)
        counts = {}
        for pos, _ in samples:
            if pos not in counts:
                counts[pos] = con.execute(
                    "SELECT COUNT(*) FROM person_templates "
                    "WHERE uid=? AND modality=? AND finger_position=?",
                    (uid, modality, pos)).fetchone()[0]
        full = sorted(p for p, n in counts.items() if n >= cap)
        if full:
            con.rollback()
            raise SampleCapExceeded(
                "finger already has the maximum number of samples",
                fingers=full, max_samples=cap)
        now = _now()
        for pos, minutiae in samples:
            con.execute(
                "INSERT INTO person_templates "
                "(uid,modality,finger_position,minutiae_json,minutiae_count,created_at) "
                "VALUES (?,?,?,?,?,?)",
                (uid, modality, pos, json.dumps(minutiae), len(minutiae), now))
            counts[pos] += 1
        con.commit()
        return counts
    except Exception:
        if con.in_transaction:
            con.rollback()
        raise
    finally:
        con.close()


def _load_samples(uid, modality):
    """{finger_position: [minutiae, ...]} for one person and modality."""
    con = _connect()
    try:
        _require_person(con, uid)
        rows = con.execute(
            "SELECT finger_position, minutiae_json FROM person_templates "
            "WHERE uid=? AND modality=? ORDER BY id", (uid, modality)).fetchall()
    finally:
        con.close()
    out = {}
    for r in rows:
        out.setdefault(r["finger_position"], []).append(json.loads(r["minutiae_json"]))
    return out


def _log_verification(uid, modality, matched, score):
    con = _connect()
    try:
        con.execute(
            "INSERT INTO person_verifications (uid,modality,matched,score,created_at) "
            "VALUES (?,?,?,?,?)", (uid, modality, 1 if matched else 0, score, _now()))
        con.commit()
    finally:
        con.close()


# ── verification ─────────────────────────────────────────────────────────────
def _best_over_samples(probe, samples):
    algo = match_algorithm()
    return max(match_templates(probe, s, algorithm=algo) for s in samples)


def verify_single(uid, finger_position, probe_minutiae):
    uid = normalize_uid(uid)
    pos = normalize_position("single", finger_position)
    stored = _load_samples(uid, "single").get(pos)
    if not stored:
        raise FingerNotEnrolled("no enrolled sample for this finger", finger_position=pos)
    score = float(_best_over_samples(probe_minutiae, stored))
    thresh = match_threshold()
    matched = score >= thresh
    _log_verification(uid, "single", matched, score)
    return {"uid": uid, "finger_position": pos, "matched": matched,
            "score": round(score, 5), "threshold": thresh,
            "samples_compared": len(stored)}


def verify_slap(uid, probe_fingers):
    """probe_fingers: [{"finger_position": str, "minutiae": [...]}, ...]"""
    uid = normalize_uid(uid)
    stored = _load_samples(uid, "slap")
    need = min_fingers_pass()
    if len(stored) < need:
        raise NotEnoughEnrolledFingers(
            "uid has fewer enrolled slap fingers than the pass rule needs",
            enrolled_fingers=len(stored), required=need)

    thresh = match_threshold()
    fingers = []
    passed = 0
    for f in probe_fingers:
        pos = normalize_position("slap", f["finger_position"])
        samples = stored.get(pos)
        if not samples:
            fingers.append({"finger_position": pos, "enrolled": False,
                            "score": None, "passed": False})
            continue
        score = float(_best_over_samples(f["minutiae"], samples))
        ok = score >= thresh
        passed += ok
        fingers.append({"finger_position": pos, "enrolled": True,
                        "score": round(score, 5), "passed": ok,
                        "samples_compared": len(samples)})

    matched = passed >= need
    scores = [x["score"] for x in fingers if x["score"] is not None]
    _log_verification(uid, "slap", matched, max(scores) if scores else None)
    return {"uid": uid, "matched": matched, "fingers_passed": passed,
            "fingers_required": need, "threshold": thresh, "fingers": fingers}


# ── HTTP plumbing shared by both servers ─────────────────────────────────────
def require_api_key(fn):
    """Reject requests without a valid X-API-Key. With API_KEY unset on the server
    the endpoint refuses to run at all, rather than silently running open."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        expected = os.environ.get("API_KEY", "")
        if not expected:
            return jsonify({"success": False, "error": "api_key_not_configured"}), 503
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            return jsonify({"success": False, "error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


def error_response(err):
    body = {"success": False, "error": err.code, "message": str(err)}
    body.update(err.extra)
    return jsonify(body), err.status


def handles_identity_errors(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except IdentityError as e:
            return error_response(e)
    return wrapper


identity_bp = Blueprint("identity", __name__)


@identity_bp.route("/uid/register", methods=["POST"])
@require_api_key
@handles_identity_errors
def uid_register():
    person = register_person(request.form.get("uid"), request.form.get("name"))
    return jsonify({"success": True, "uid": person["uid"], "name": person["name"],
                    "created_at": person["created_at"]}), 201


@identity_bp.route("/uid/<uid>", methods=["PATCH"])
@require_api_key
@handles_identity_errors
def uid_rename(uid):
    return jsonify({"success": True, **rename_person(uid, request.form.get("name"))})


@identity_bp.route("/uid/<uid>", methods=["DELETE"])
@require_api_key
@handles_identity_errors
def uid_delete(uid):
    # Guard against stray calls: the caller must explicitly confirm.
    if request.args.get("confirm", request.form.get("confirm", "")).lower() != "true":
        raise BadRequest("confirm=true is required to delete a person")
    removed = delete_person(uid)
    return jsonify({"success": True, "uid": normalize_uid(uid),
                    "deleted_samples": removed})


@identity_bp.route("/uid/<uid>/samples", methods=["DELETE"])
@require_api_key
@handles_identity_errors
def uid_delete_finger(uid):
    n = delete_finger(uid, request.args.get("modality", request.form.get("modality", "")),
                      request.args.get("finger_position", request.form.get("finger_position")))
    return jsonify({"success": True, "uid": normalize_uid(uid), "deleted_samples": n})


@identity_bp.route("/uid/<uid>", methods=["GET"])
@require_api_key
@handles_identity_errors
def uid_info(uid):
    return jsonify({"success": True, **enrolled_summary(uid)})
