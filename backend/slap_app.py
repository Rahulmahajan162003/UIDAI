"""
slap_app.py — Multi-finger (slap) contactless fingerprint REST service.

Self-contained in THIS folder; does not touch uidai_app. Uses slap_core (which
holds faithful copies of the proven single-finger pipeline) and loops it over
EVERY detected finger.

Per request:
    CAPTURE     -> slap_core.detect_all_finger_boxes  (all fingers, not argmax)
    PRE-PROCESS -> slap_core.preprocess_fingerprint   (U2-Net + Zero-DCE/hist-eq)
    MINUTIAE    -> slap_core.detect_minutiae          (MinutiaeNet)
    TEMPLATE    -> slap_core.export_iso_template       (ISO 19794-2 FMR, per finger)

Run:
    pip install -r requirements.txt
    python slap_app.py            # -> http://0.0.0.0:5010
"""

import os
import json
import base64
import sqlite3
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify

import slap_core as core
import fir  # same module app.py's /export_fir already uses
from spoof_detection import detect_screen_replay  # same module app.py uses

app = Flask(__name__)

SLAP_DB = os.environ.get(
    "SLAP_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "slap.db"),
)

# ISO 19794-2 finger position codes.
ISO_FINGER_CODES = {
    "right_thumb": 1, "right_index": 2, "right_middle": 3, "right_ring": 4, "right_little": 5,
    "left_thumb": 6, "left_index": 7, "left_middle": 8, "left_ring": 9, "left_little": 10,
}

# A 4-finger slap (no thumb), labelled by LEFT->RIGHT image order. This is
# orientation dependent — override per request with form field `finger_order`
# (comma-separated) if your capture geometry differs.
SLAP_ORDER = {
    "right": ["right_little", "right_ring", "right_middle", "right_index"],
    "left":  ["left_index", "left_middle", "left_ring", "left_little"],
}


def init_slap_db():
    con = sqlite3.connect(SLAP_DB)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS slap_templates (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            uid             TEXT NOT NULL,
            name            TEXT,
            batch           TEXT NOT NULL,
            finger_position TEXT NOT NULL,
            iso_code        INTEGER,
            minutiae_count  INTEGER,
            template_b64    TEXT NOT NULL,
            minutiae_json   TEXT,
            created_at      TEXT NOT NULL,
            UNIQUE(uid, batch, finger_position)
        )
        """
    )
    con.commit()
    con.close()


def label_fingers(dets, hand_side, finger_order=None):
    order = finger_order or SLAP_ORDER.get(hand_side, [])
    for idx, d in enumerate(dets):
        pos = order[idx] if idx < len(order) else f"{hand_side}_finger_{idx + 1}"
        d["finger_position"] = pos
        d["iso_code"] = ISO_FINGER_CODES.get(pos)
    return dets


def process_one_finger(crop_bgr, want_vis=False):
    """ONE cropped finger through the proven preprocess -> minutiae -> ISO chain."""
    liveness     = core.check_liveness(crop_bgr)
    screen_check = detect_screen_replay(crop_bgr)
    mask         = core.get_segmentation_mask(crop_bgr)
    preprocessed = core.preprocess_fingerprint(crop_bgr, mask=mask)
    minutiae     = core.detect_minutiae(preprocessed)
    iso          = core.export_iso_template(minutiae)

    out = {
        "liveness": liveness,
        "screen_check": screen_check,
        "minutiae_count": len(minutiae),
        "minutiae": minutiae,
        "template_b64": base64.b64encode(iso).decode(),
    }
    if want_vis:
        vis = core.create_visualization(preprocessed, minutiae)
        out["cropped_b64"]       = core.img_to_b64(crop_bgr)       # stage 1: capture
        out["preprocessed_b64"]  = core.img_to_b64(preprocessed)   # stage 2: preprocess
        out["visualization_b64"] = core.img_to_b64(vis)            # stage 3: minutiae
        out["_preprocessed_arr"] = preprocessed  # raw, for the composite (popped later)
    return out


def build_composite(image_bgr, placed):
    """Paste each preprocessed finger back at its detected bbox on a white canvas
    the size of the original capture — a 'contact-like' preprocessed slap."""
    h, w = image_bgr.shape[:2]
    canvas = np.full((h, w), 255, np.uint8)
    for (x1, y1, x2, y2), pre in placed:
        if pre is None:
            continue
        tw = x2 - x1
        ph, pw = pre.shape[:2]
        if pw != tw and pw > 0:                       # keep finger width = bbox width
            pre = cv2.resize(pre, (tw, max(1, int(ph * tw / pw))))
            ph, pw = pre.shape[:2]
        y_end = min(y1 + ph, h)
        x_end = min(x1 + pw, w)
        canvas[y1:y_end, x1:x_end] = pre[: y_end - y1, : x_end - x1]
    return canvas


def process_slap(image_bgr, hand_side="right", conf=0.25, finger_order=None, want_vis=False):
    dets = core.detect_all_finger_boxes(image_bgr, conf=conf)
    if not dets:
        return {"finger_count": 0, "fingers": [], "error": "No fingers detected"}
    dets = label_fingers(dets, hand_side, finger_order=finger_order)

    fingers = []
    placed = []  # (bbox, preprocessed_arr) for the composite
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        crop = image_bgr[y1:y2, x1:x2]
        finger = {
            "finger_position": d["finger_position"],
            "iso_code": d["iso_code"],
            "detection_conf": round(d["conf"], 4),
            "detected_class": d.get("cls_name"),
            "bbox": [x1, y1, x2, y2],
        }
        try:
            res = process_one_finger(crop, want_vis=want_vis)
            placed.append(((x1, y1, x2, y2), res.pop("_preprocessed_arr", None)))
            finger.update(res)
            finger["ok"] = True
        except Exception as e:  # one bad finger must not kill the whole slap
            finger["ok"] = False
            finger["error"] = str(e)
        fingers.append(finger)

    out = {"finger_count": len(fingers), "hand_side": hand_side, "fingers": fingers}
    if want_vis and placed:
        out["composite_b64"] = core.img_to_b64(build_composite(image_bgr, placed))
    return out


def _read_image():
    if "image" not in request.files:
        return None
    data = np.frombuffer(request.files["image"].read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _parse_finger_order():
    raw = request.form.get("finger_order", "").strip()
    return [p.strip() for p in raw.split(",") if p.strip()] or None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "slap-multi-finger",
        "models_loaded": list(core.global_models.keys()),
        "device": str(core.DEVICE),
    })


@app.route("/quality_check", methods=["POST"])
def quality_check_endpoint():
    """Fast gate used by the camera widget's auto-capture poll. Reports whether
    finger(s) are present + basic blur/brightness so the UI can hold-and-commit.
    Returns the same shape the FingerprintCameraWidget expects."""
    img = _read_image()
    if img is None:
        return jsonify({"passed": False, "finger_detected": False,
                        "guidance": "No image"}), 400

    dets = core.detect_all_finger_boxes(img, conf=0.25)
    n = len(dets)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = blur_score < 60.0
    brightness = float(gray.mean())
    too_dark = brightness < 55.0
    too_bright = brightness > 205.0

    # Pixel-overexposure glare check — same fallback formula single-finger's
    # check_glare() uses when no glare model is available; slap has no glare model
    # at all, so this is the only signal, not a fallback from a model attempt.
    overexposed_ratio = float(np.mean(gray > 240))
    has_glare = overexposed_ratio > 0.05

    # ROI check — every detected finger must sit inside the frame with a small
    # margin, not clipped at the edge. Previously hardcoded to True/no guidance.
    h, w = img.shape[:2]
    margin_x, margin_y = int(w * 0.02), int(h * 0.02)
    in_roi = bool(dets) and all(
        d["bbox"][0] > margin_x and d["bbox"][1] > margin_y and
        d["bbox"][2] < w - margin_x and d["bbox"][3] < h - margin_y
        for d in dets
    )
    roi_guidance = "" if in_roi else "Move hand toward center — finger(s) too close to frame edge"

    if n == 0:
        guidance = "Place your hand in the frame"
    elif is_blurry:
        guidance = "Hold steady — image blurry"
    elif too_dark:
        guidance = "Too dark — add light / torch"
    elif too_bright:
        guidance = "Too bright — reduce glare"
    elif has_glare:
        guidance = "Glare detected — reduce direct light"
    elif not in_roi:
        guidance = roi_guidance
    else:
        guidance = f"{n} finger{'s' if n != 1 else ''} detected — hold still"

    passed = ((n >= 1) and (not is_blurry) and (not too_dark) and (not too_bright)
              and (not has_glare) and in_roi)
    return jsonify({
        "passed": passed,
        "finger_detected": n > 0,
        "finger_count": n,
        "detection_conf": round(dets[0]["conf"], 4) if dets else 0.0,
        "guidance": guidance,
        "blur": {"blur_score": round(blur_score, 1), "is_blurry": is_blurry},
        "brightness": {"brightness": round(brightness, 1),
                       "too_dark": too_dark, "too_bright": too_bright},
        "glare": {"has_glare": has_glare,
                  "overexposed_ratio": round(overexposed_ratio, 4)},
        "in_roi": in_roi,
        "roi_guidance": roi_guidance,
    })


@app.route("/process_slap", methods=["POST"])
def process_slap_endpoint():
    """Detect + preprocess + minutiae for ALL fingers (no DB write).
    Fields: image(file), hand_side=left|right, conf=float, vis=0|1, finger_order."""
    img = _read_image()
    if img is None:
        return jsonify({"error": "No image"}), 400
    result = process_slap(
        img,
        hand_side=request.form.get("hand_side", "right"),
        conf=float(request.form.get("conf", 0.25)),
        finger_order=_parse_finger_order(),
        want_vis=request.form.get("vis", "0") in ("1", "true", "True"),
    )
    return jsonify(result)


@app.route("/enroll_slap", methods=["POST"])
def enroll_slap_endpoint():
    """Store one ISO 19794-2 template PER detected finger.
    Fields: image(file), name, uid, batch, hand_side, finger_order."""
    img = _read_image()
    if img is None:
        return jsonify({"error": "No image"}), 400
    uid = request.form.get("uid", "")
    if not uid:
        return jsonify({"error": "uid required"}), 400
    name      = request.form.get("name", "")
    batch     = request.form.get("batch", "default")
    hand_side = request.form.get("hand_side", "right")

    result = process_slap(img, hand_side=hand_side, finger_order=_parse_finger_order())
    if result["finger_count"] == 0:
        return jsonify(result), 422

    # Fail closed on liveness, same as authenticate_slap/verify_slap — a spoofed
    # capture must not be enrollable just because detection+minutiae succeeded.
    for f in result["fingers"]:
        if not f.get("ok"):
            continue
        lv = f.get("liveness") or {}
        sc = f.get("screen_check") or {}
        if not lv.get("is_live") or sc.get("is_screen"):
            return jsonify({
                "success": False,
                "spoof_detected": True,
                "finger_position": f.get("finger_position"),
                "liveness_confidence": lv.get("confidence", 0.0),
                "screen_check": sc,
            }), 422

    con = sqlite3.connect(SLAP_DB)
    now = datetime.utcnow().isoformat()
    saved = []
    for f in result["fingers"]:
        if not f.get("ok"):
            continue
        con.execute(
            """INSERT OR REPLACE INTO slap_templates
               (uid,name,batch,finger_position,iso_code,minutiae_count,
                template_b64,minutiae_json,created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (uid, name, batch, f["finger_position"], f["iso_code"],
             f["minutiae_count"], f["template_b64"], json.dumps(f["minutiae"]), now),
        )
        saved.append({
            "finger_position": f["finger_position"],
            "iso_code": f["iso_code"],
            "minutiae_count": f["minutiae_count"],
        })
    con.commit()
    con.close()

    return jsonify({
        "uid": uid, "name": name, "batch": batch, "hand_side": hand_side,
        "enrolled_fingers": saved, "finger_count": len(saved),
    })


# ══════════════════════════════════════════════════════════════════════════════
# MATCHING ENGINE (imported + MINUTIAE MCC Graph + Relaxation Labeling)
# ══════════════════════════════════════════════════════════════════════════════
import traceback

# Shared slap aggregation — see slap_matching.py (which itself imports
# matching.py). This file used to have its own local _best_finger_match()
# copy; consolidated so slap matching genuinely lives in one place, not
# copies that can drift apart again (see bug.md #8, which was exactly that
# happening).
from slap_matching import score_against_user

MATCH_THRESH  = 0.20


def init_slap_history_db():
    con = sqlite3.connect(SLAP_DB)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS slap_auth_history (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            uid                  TEXT,
            name                 TEXT,
            batch                TEXT NOT NULL,
            hand_side            TEXT,
            success              INTEGER NOT NULL,
            avg_confidence       REAL,
            best_confidence      REAL,
            matched_fingers_json TEXT,
            created_at           TEXT NOT NULL
        )
        """
    )
    con.commit()
    con.close()


# ── Authenticate (1:N) slap ──────────────────────────────────────────────────
@app.route("/authenticate_slap", methods=["POST"])
def authenticate_slap_endpoint():
    try:
        batch = request.form.get("batch", "").strip()
        hand_side = request.form.get("hand_side", "right")
        if not batch:
            return jsonify({"success": False, "error": "batch required"}), 400
        img = _read_image()
        if img is None:
            return jsonify({"success": False, "error": "image required"}), 400

        result = process_slap(img, hand_side=hand_side,
                              finger_order=_parse_finger_order())
        if result["finger_count"] == 0:
            return jsonify({"success": False, "quality_failed": True,
                            "guidance": "No fingers detected",
                            "finger_count": 0}), 422

        fingers = [f for f in result["fingers"] if f.get("ok")]
        for f in fingers:
            lv = f.get("liveness") or {}
            sc = f.get("screen_check") or {}
            if not lv.get("is_live") or sc.get("is_screen"):
                return jsonify({
                    "success": False,
                    "spoof_detected": True,
                    "finger_position": f.get("finger_position"),
                    "liveness_confidence": lv.get("confidence", 0.0),
                    "screen_check": sc,
                }), 422

        con = sqlite3.connect(SLAP_DB)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT uid, name, finger_position, minutiae_json "
                "FROM slap_templates WHERE batch = ? ORDER BY uid",
                (batch,),
            ).fetchall()
            if not rows:
                return jsonify({"success": False,
                                "message": "No enrolled slap users in batch"}), 404

            # group rows per uid
            by_uid = {}
            for r in rows:
                by_uid.setdefault(r["uid"], []).append(
                    {"finger_position": r["finger_position"],
                     "name": r["name"],
                     "minutiae_json": r["minutiae_json"]})

            # Shared aggregation from slap_matching.py — see the note on
            # authenticate_slap_v2 / verify_slap_v2; consolidating this copy
            # too so there is genuinely one implementation, not three.
            best_uid = None
            best_name = None
            best_agg = 0.0
            best_matches = []
            per_user = []
            for uid, user_rows in by_uid.items():
                agg_score, finger_scores = score_against_user(fingers, user_rows)
                if not finger_scores:
                    continue
                per_user.append({"uid": uid, "agg_score": round(agg_score, 5),
                                 "matched_fingers": finger_scores})
                if agg_score > best_agg:
                    best_agg = agg_score
                    best_uid = uid
                    best_name = user_rows[0]["name"]
                    best_matches = finger_scores

            success = best_agg >= MATCH_THRESH and best_uid is not None
            now = datetime.utcnow().isoformat()
            if success:
                avg_conf = (sum(m["confidence"] for m in best_matches)
                            / max(1, len(best_matches)))
                con.execute(
                    """INSERT INTO slap_auth_history
                    (uid,name,batch,hand_side,success,avg_confidence,best_confidence,
                     matched_fingers_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (best_uid, best_name, batch, hand_side, 1,
                     round(avg_conf, 5), round(best_agg, 5),
                     json.dumps(best_matches), now),
                )
                con.commit()
                return jsonify({
                    "success": True,
                    "uid": best_uid,
                    "name": best_name,
                    "hand_side": hand_side,
                    "avg_confidence": round(avg_conf, 5),
                    "confidence": round(best_agg, 5),
                    "threshold": MATCH_THRESH,
                    "matched_fingers": best_matches,
                })
            else:
                con.execute(
                    """INSERT INTO slap_auth_history
                    (uid,name,batch,hand_side,success,avg_confidence,best_confidence,
                     matched_fingers_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (None, None, batch, hand_side, 0, 0.0, round(best_agg, 5),
                     json.dumps(per_user[:10]), now),
                )
                con.commit()
                return jsonify({
                    "success": False,
                    "message": "No match found",
                    "confidence": round(best_agg, 5),
                    "threshold": MATCH_THRESH,
                })
        finally:
            con.close()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── Verify (1:1) slap ────────────────────────────────────────────────────────
@app.route("/verify_slap", methods=["POST"])
def verify_slap_endpoint():
    try:
        uid = request.form.get("uid", "").strip()
        batch = request.form.get("batch", "").strip()
        hand_side = request.form.get("hand_side", "right")
        if not uid or not batch:
            return jsonify({"success": False,
                            "error": "uid and batch required"}), 400
        img = _read_image()
        if img is None:
            return jsonify({"success": False, "error": "image required"}), 400

        result = process_slap(img, hand_side=hand_side,
                              finger_order=_parse_finger_order())
        if result["finger_count"] == 0:
            return jsonify({"success": False, "quality_failed": True,
                            "guidance": "No fingers detected"}), 422

        fingers = [f for f in result["fingers"] if f.get("ok")]
        for f in fingers:
            lv = f.get("liveness") or {}
            sc = f.get("screen_check") or {}
            if not lv.get("is_live") or sc.get("is_screen"):
                return jsonify({
                    "success": False,
                    "spoof_detected": True,
                    "finger_position": f.get("finger_position"),
                    "liveness_confidence": lv.get("confidence", 0.0),
                    "screen_check": sc,
                }), 422

        con = sqlite3.connect(SLAP_DB)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT uid, name, finger_position, minutiae_json "
                "FROM slap_templates WHERE uid=? AND batch=?",
                (uid, batch),
            ).fetchall()
            if not rows:
                return jsonify({"success": False,
                                "error": f"No slap enrollment for uid={uid}"}), 404
            name = rows[0]["name"]

            # Shared aggregation from slap_matching.py — was duplicated here
            # using `all_scores` (includes zero-scoring fingers before taking
            # top-2), which silently dilutes a genuine match whenever any
            # presented finger doesn't match anything (bug.md #8). Fixed to
            # use the same consolidated logic authenticate_slap_v2 now uses.
            user_rows = [{"finger_position": r["finger_position"],
                         "minutiae_json": r["minutiae_json"]} for r in rows]
            score, matched_fingers = score_against_user(fingers, user_rows)
            matched = score >= MATCH_THRESH
            avg_conf = (sum(m["confidence"] for m in matched_fingers)
                        / max(1, len(matched_fingers)))

            return jsonify({
                "success": True,
                "matched": matched,
                "uid": uid,
                "name": name,
                "hand_side": hand_side,
                "confidence": round(score, 5),
                "avg_confidence": round(avg_conf, 5),
                "threshold": MATCH_THRESH,
                "matched_fingers": matched_fingers,
            })
        finally:
            con.close()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# V2 — CLOUD-ONLY SLAP ENDPOINTS (templatization + matching)
#
# Mirrors the split in app.py's /v2/enroll, /v2/authenticate, /v2/verify: on-device
# now owns per-finger detection, quality control, liveness, and preprocessing
# (segmentation + enhancement) for the whole slap. These endpoints take the
# already-preprocessed per-finger images directly and skip straight to
# templatization (core.detect_minutiae) and matching — no detect_all_finger_boxes,
# no liveness, no segmentation/enhancement here, that already happened on-device.
#
# Upload shape: repeated `image` file fields, one per finger, in left-to-right
# capture order — same convention detect_all_finger_boxes + label_fingers() already
# use today (via hand_side / finger_order), just driven by upload order instead of
# on-server detection order.
#
# Additive only: /process_slap, /enroll_slap, /authenticate_slap, /verify_slap
# above are untouched, so existing clients keep working unmodified.
# ══════════════════════════════════════════════════════════════════════════════

def _read_images_list(field_name="image"):
    imgs = []
    for f in request.files.getlist(field_name):
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
    return imgs


def _label_by_order(n, hand_side, finger_order=None):
    """Assign finger_position/iso_code by upload order — the on-device-driven
    equivalent of label_fingers(), which assigns by detected-bbox left-to-right
    order."""
    order = finger_order or SLAP_ORDER.get(hand_side, [])
    labels = []
    for idx in range(n):
        pos = order[idx] if idx < len(order) else f"{hand_side}_finger_{idx + 1}"
        labels.append({"finger_position": pos, "iso_code": ISO_FINGER_CODES.get(pos)})
    return labels


@app.route("/v2/enroll_slap", methods=["POST"])
def enroll_slap_v2():
    try:
        uid = request.form.get("uid", "").strip()
        if not uid:
            return jsonify({"success": False, "error": "uid required"}), 400
        name      = request.form.get("name", "")
        batch     = request.form.get("batch", "default")
        hand_side = request.form.get("hand_side", "right")

        imgs = _read_images_list()
        if not imgs:
            return jsonify({"success": False, "error": "at least one image required"}), 400
        labels = _label_by_order(len(imgs), hand_side, _parse_finger_order())

        con = sqlite3.connect(SLAP_DB)
        now = datetime.utcnow().isoformat()
        saved = []
        for img, label in zip(imgs, labels):
            minutiae = core.detect_minutiae(img)
            iso = core.export_iso_template(minutiae)
            con.execute(
                """INSERT OR REPLACE INTO slap_templates
                   (uid,name,batch,finger_position,iso_code,minutiae_count,
                    template_b64,minutiae_json,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (uid, name, batch, label["finger_position"], label["iso_code"],
                 len(minutiae), base64.b64encode(iso).decode(),
                 json.dumps(minutiae), now),
            )
            saved.append({
                "finger_position": label["finger_position"],
                "iso_code": label["iso_code"],
                "minutiae_count": len(minutiae),
            })
        con.commit()
        con.close()

        return jsonify({
            "success": True,
            "uid": uid, "name": name, "batch": batch, "hand_side": hand_side,
            "enrolled_fingers": saved, "finger_count": len(saved),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/v2/authenticate_slap", methods=["POST"])
def authenticate_slap_v2():
    try:
        batch = request.form.get("batch", "").strip()
        hand_side = request.form.get("hand_side", "right")
        if not batch:
            return jsonify({"success": False, "error": "batch required"}), 400

        imgs = _read_images_list()
        if not imgs:
            return jsonify({"success": False, "error": "at least one image required"}), 400
        labels = _label_by_order(len(imgs), hand_side, _parse_finger_order())
        fingers = [{"finger_position": lbl["finger_position"],
                    "minutiae": core.detect_minutiae(img)}
                   for img, lbl in zip(imgs, labels)]

        con = sqlite3.connect(SLAP_DB)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT uid, name, finger_position, minutiae_json "
                "FROM slap_templates WHERE batch = ? ORDER BY uid",
                (batch,),
            ).fetchall()
            if not rows:
                return jsonify({"success": False,
                                "message": "No enrolled slap users in batch"}), 404

            by_uid = {}
            for r in rows:
                by_uid.setdefault(r["uid"], []).append(
                    {"finger_position": r["finger_position"],
                     "name": r["name"],
                     "minutiae_json": r["minutiae_json"]})

            # Shared aggregation from slap_matching.py — was duplicated here as
            # its own, still-uses-_best_finger_match-only loop, missing the
            # top-2-of-nonzero-scores consolidation (bug.md #8) even though
            # that fix was written this same session. Fixed to actually use it.
            best_uid = None
            best_name = None
            best_agg = 0.0
            best_matches = []
            per_user = []
            for uid, user_rows in by_uid.items():
                agg_score, finger_scores = score_against_user(fingers, user_rows)
                if not finger_scores:
                    continue
                per_user.append({"uid": uid, "agg_score": round(agg_score, 5),
                                 "matched_fingers": finger_scores})
                if agg_score > best_agg:
                    best_agg = agg_score
                    best_uid = uid
                    best_name = user_rows[0]["name"]
                    best_matches = finger_scores

            success = best_agg >= MATCH_THRESH and best_uid is not None
            now = datetime.utcnow().isoformat()
            if success:
                avg_conf = (sum(m["confidence"] for m in best_matches)
                            / max(1, len(best_matches)))
                con.execute(
                    """INSERT INTO slap_auth_history
                    (uid,name,batch,hand_side,success,avg_confidence,best_confidence,
                     matched_fingers_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (best_uid, best_name, batch, hand_side, 1,
                     round(avg_conf, 5), round(best_agg, 5),
                     json.dumps(best_matches), now),
                )
                con.commit()
                return jsonify({
                    "success": True,
                    "uid": best_uid,
                    "name": best_name,
                    "hand_side": hand_side,
                    "avg_confidence": round(avg_conf, 5),
                    "confidence": round(best_agg, 5),
                    "threshold": MATCH_THRESH,
                    "matched_fingers": best_matches,
                })
            else:
                con.execute(
                    """INSERT INTO slap_auth_history
                    (uid,name,batch,hand_side,success,avg_confidence,best_confidence,
                     matched_fingers_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (None, None, batch, hand_side, 0, 0.0, round(best_agg, 5),
                     json.dumps(per_user[:10]), now),
                )
                con.commit()
                return jsonify({
                    "success": False,
                    "message": "No match found",
                    "confidence": round(best_agg, 5),
                    "threshold": MATCH_THRESH,
                })
        finally:
            con.close()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/v2/verify_slap", methods=["POST"])
def verify_slap_v2():
    try:
        uid = request.form.get("uid", "").strip()
        batch = request.form.get("batch", "").strip()
        hand_side = request.form.get("hand_side", "right")
        if not uid or not batch:
            return jsonify({"success": False,
                            "error": "uid and batch required"}), 400

        imgs = _read_images_list()
        if not imgs:
            return jsonify({"success": False, "error": "at least one image required"}), 400
        labels = _label_by_order(len(imgs), hand_side, _parse_finger_order())
        fingers = [{"finger_position": lbl["finger_position"],
                    "minutiae": core.detect_minutiae(img)}
                   for img, lbl in zip(imgs, labels)]

        con = sqlite3.connect(SLAP_DB)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT uid, name, finger_position, minutiae_json "
                "FROM slap_templates WHERE uid=? AND batch=?",
                (uid, batch),
            ).fetchall()
            if not rows:
                return jsonify({"success": False,
                                "error": f"No slap enrollment for uid={uid}"}), 404
            name = rows[0]["name"]

            # Shared aggregation from slap_matching.py — was duplicated here
            # using `all_scores` (includes zero-scoring fingers before taking
            # top-2), which silently dilutes a genuine match whenever any
            # presented finger doesn't match anything (bug.md #8). Fixed to
            # use the same consolidated logic authenticate_slap_v2 now uses.
            user_rows = [{"finger_position": r["finger_position"],
                         "minutiae_json": r["minutiae_json"]} for r in rows]
            score, matched_fingers = score_against_user(fingers, user_rows)
            matched = score >= MATCH_THRESH
            avg_conf = (sum(m["confidence"] for m in matched_fingers)
                        / max(1, len(matched_fingers)))

            return jsonify({
                "success": True,
                "matched": matched,
                "uid": uid,
                "name": name,
                "hand_side": hand_side,
                "confidence": round(score, 5),
                "avg_confidence": round(avg_conf, 5),
                "threshold": MATCH_THRESH,
                "matched_fingers": matched_fingers,
            })
        finally:
            con.close()
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── Export FIR (per finger) ──────────────────────────────────────────────────
@app.route("/export_fir_slap", methods=["POST"])
def export_fir_slap():
    """
    Produce one ISO/IEC 19794-4 FIR per detected finger.

    Mirrors app.py's /export_fir exactly (detect+crop -> preprocess -> normalise to
    500 DPI -> encode -> self-decode to prove it round-trips), just looped over
    every finger in the slap instead of one. fir.encode_fir()'s finger_position
    already uses the standard's 0-10 codes, which is exactly what label_fingers()'s
    iso_code already computes for each detected finger — no remapping needed.

    fir.py's own record format is single-finger (number_of_fingers hardcoded to 1),
    so "a slap FIR" here means N independent standard-conformant records, one per
    finger, not one record carrying four fingers — that keeps the byte format
    untouched rather than extending a standard's on-wire layout speculatively.

    Fields: image (file), hand_side=left|right, finger_order (optional), source_dpi
    (optional).
    """
    try:
        img = _read_image()
        if img is None:
            return jsonify({"success": False, "error": "image required"}), 400

        hand_side = request.form.get("hand_side", "right")
        source_dpi = request.form.get("source_dpi")
        source_dpi = int(source_dpi) if source_dpi else None

        dets = core.detect_all_finger_boxes(img, conf=0.25)
        if not dets:
            return jsonify({"success": False, "error": "No fingers detected"}), 422
        dets = label_fingers(dets, hand_side, finger_order=_parse_finger_order())

        records = []
        errors = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            crop = img[y1:y2, x1:x2]
            try:
                mask = core.get_segmentation_mask(crop)
                preprocessed = core.preprocess_fingerprint(crop, mask=mask)

                normalised, norm_info = fir.normalize_to_500dpi(preprocessed,
                                                                 source_dpi=source_dpi)
                # iso_code is None for a detected finger beyond the 4-slot
                # SLAP_ORDER/finger_order list (a 5th blob, stray detection,
                # etc.) — fir.py's own FINGER_POSITIONS defines 0 as "unknown"
                # for exactly this case; fall back to it instead of passing
                # None into encode_fir()'s int range check, which would raise
                # a raw TypeError instead of a clean, expected FIRError.
                fir_position = d["iso_code"] if d["iso_code"] is not None else 0
                record = fir.encode_fir(normalised, finger_position=fir_position)
                # Decode our own output before returning it, same as /export_fir —
                # a record that will not parse is worse than no record.
                parsed = fir.decode_fir(record)

                records.append({
                    "finger_position": d["finger_position"],
                    "iso_code": d["iso_code"],
                    "fir_base64": base64.b64encode(record).decode(),
                    "record_bytes": len(record),
                    "width": parsed["finger"]["width"],
                    "height": parsed["finger"]["height"],
                    "resolution_dpi": parsed["header"]["image_resolution_x"],
                    "normalisation": norm_info,
                })
            except fir.FIRError as e:
                errors.append({"finger_position": d["finger_position"],
                               "error": "FIR encoding failed: %s" % e})
            except Exception as e:
                errors.append({"finger_position": d["finger_position"], "error": str(e)})

        return jsonify({
            "success": len(records) > 0,
            "hand_side": hand_side,
            "finger_count": len(records),
            "records": records,
            "errors": errors,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ── Slap Auth History ────────────────────────────────────────────────────────
@app.route("/history", methods=["GET"])
def history_endpoint():
    batch = request.args.get("batch", "").strip()
    con = sqlite3.connect(SLAP_DB)
    try:
        c = con.cursor()
        c.execute(
            """
            SELECT uid, name, hand_side, success, avg_confidence,
                   best_confidence, matched_fingers_json, created_at
            FROM slap_auth_history
            WHERE (?='' OR batch=?)
            ORDER BY created_at DESC LIMIT 100
            """,
            (batch, batch),
        )
        rows = c.fetchall()
    finally:
        con.close()

    history = []
    for r in rows:
        try:
            matched = json.loads(r[6]) if r[6] else []
        except Exception:
            matched = []
        history.append({
            "type": "slap",
            "uid": r[0],
            "name": r[1],
            "hand_side": r[2],
            "success": bool(r[3]),
            "avg_confidence": r[4],
            "confidence": r[5] or r[4],
            "matched_fingers": matched,
            "finger_count": len(matched) if isinstance(matched, list) else 0,
            "timestamp": r[7],
        })
    return jsonify({"history": history})


# ── UID-keyed slap enrol / 1:1 verify ────────────────────────────────────────
# Person-centric flow shared with the single-finger server via identity_store
# (identity.db): one 12-char UID per person. Independent of the batch-keyed
# endpoints above.
import identity_store as ids

app.register_blueprint(ids.identity_bp)   # /uid/register, GET /uid/<uid>
ids.init_identity_db()


def _uid_slap_fingers():
    """Run the slap pipeline on the uploaded image and apply the spoof gate.

    Returns (fingers, None) with fingers = [{finger_position, minutiae, ...}] for
    every finger that processed OK and has enough minutiae, or (None, response)."""
    img = _read_image()
    if img is None:
        return None, (jsonify({"success": False, "error": "image required"}), 400)
    result = process_slap(img, hand_side=request.form.get("hand_side", "right"),
                          finger_order=_parse_finger_order())
    if result["finger_count"] == 0:
        return None, (jsonify({"success": False, "quality_failed": True,
                               "guidance": "No fingers detected",
                               "finger_count": 0}), 422)
    ok = [f for f in result["fingers"] if f.get("ok")]
    for f in ok:
        lv = f.get("liveness") or {}
        sc = f.get("screen_check") or {}
        if not lv.get("is_live") or sc.get("is_screen"):
            return None, (jsonify({
                "success": False, "spoof_detected": True,
                "finger_position": f.get("finger_position"),
                "liveness_confidence": lv.get("confidence", 0.0),
                "screen_check": sc}), 422)
    usable = [f for f in ok if len(f["minutiae"]) >= ids.MIN_MINUTIAE]
    skipped = [f["finger_position"] for f in result["fingers"] if f not in usable]
    if not usable:
        return None, (jsonify({"success": False, "error": "Too few minutiae — retake",
                               "skipped_fingers": skipped}), 422)
    return (usable, skipped), None


@app.route("/uid/enroll_slap", methods=["POST"])
@ids.require_api_key
@ids.handles_identity_errors
def uid_enroll_slap():
    try:
        uid = ids.normalize_uid(request.form.get("uid"))
        ids.enrolled_summary(uid)          # 404 before spending time on the pipeline
        got, resp = _uid_slap_fingers()
        if resp is not None:
            return resp
        usable, skipped = got
        counts = ids.add_samples(
            uid, "slap", [(f["finger_position"], f["minutiae"]) for f in usable])
        return jsonify({
            "success": True, "uid": uid, "max_samples": ids.max_samples(),
            "enrolled_fingers": [
                {"finger_position": p, "samples_for_finger": n,
                 "minutiae_count": next(len(f["minutiae"]) for f in usable
                                        if ids.normalize_position("slap", f["finger_position"]) == p)}
                for p, n in counts.items()],
            "skipped_fingers": skipped})
    except ids.IdentityError:
        raise
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/uid/verify_slap", methods=["POST"])
@ids.require_api_key
@ids.handles_identity_errors
def uid_verify_slap():
    try:
        uid = ids.normalize_uid(request.form.get("uid"))
        ids.enrolled_summary(uid)          # 404 before spending time on the pipeline
        got, resp = _uid_slap_fingers()
        if resp is not None:
            return resp
        usable, skipped = got
        result = ids.verify_slap(
            uid, [{"finger_position": f["finger_position"], "minutiae": f["minutiae"]}
                  for f in usable])
        return jsonify({"success": True, **result, "skipped_fingers": skipped})
    except ids.IdentityError:
        raise
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# Module-level, not inside __main__: gunicorn/wsgi.py imports this module as
# "slap_app:app" rather than executing it directly, so anything gated behind
# __main__ silently never runs under a production WSGI server. Confirmed the
# hard way — wsgi.py calls init_slap_db() itself but not this one, since
# slap_auth_history didn't exist yet when wsgi.py was written (predates
# authenticate_slap/verify_slap). Idempotent (CREATE TABLE IF NOT EXISTS), so
# safe to call unconditionally regardless of what any given entrypoint does.
init_slap_db()
init_slap_history_db()

if __name__ == "__main__":
    core.load_models()
    print("=" * 60)
    print("  Multi-finger (slap) pipeline")
    print("  POST /process_slap    — detect+preprocess+minutiae, ALL fingers")
    print("  POST /enroll_slap     — store one ISO 19794-2 template per finger")
    print("  POST /authenticate_slap — 1:N aggregate matching")
    print("  POST /verify_slap     — 1:1 matching against specific UID")
    print("  GET  /history         — slap auth history")
    print("  POST /uid/register    — register a 12-char UID")
    print("  POST /uid/enroll_slap — add slap samples under a UID")
    print("  POST /uid/verify_slap — 1:1 slap verify against a UID (2 fingers must pass)")
    print("  GET  /uid/<uid>       — enrolled finger counts")
    print("  GET  /health")
    print("=" * 60)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5010)), debug=False)

