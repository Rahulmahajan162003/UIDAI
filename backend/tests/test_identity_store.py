"""
Tests for the UID-keyed enrolment / 1:1 verification layer.

No weights, no TensorFlow, no real database — a temporary SQLite file and a
minimal Flask app carrying only the shared blueprint. Minutiae are synthetic,
same approach as test_matching.py.
"""

import math
import os
import random
import sys

import pytest
from flask import Flask

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import identity_store as ids  # noqa: E402


# ── synthetic minutiae ───────────────────────────────────────────────────────
def make_template(n=40, seed=0):
    rng = random.Random(seed)
    return [{"x": rng.uniform(0, 200), "y": rng.uniform(0, 250),
             "direction": rng.uniform(0, 2 * math.pi),
             "type": "BIF" if rng.random() > 0.5 else "RIG"} for _ in range(n)]


def jitter(tmpl, px=2.0, rad=0.05, seed=99):
    rng = random.Random(seed)
    return [{**m, "x": m["x"] + rng.uniform(-px, px), "y": m["y"] + rng.uniform(-px, px),
             "direction": (m["direction"] + rng.uniform(-rad, rad)) % (2 * math.pi)}
            for m in tmpl]


UID = "AB12CD34EF56"


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setenv("IDENTITY_DB", str(tmp_path / "identity.db"))
    monkeypatch.setenv("API_KEY", "secret")
    for var in ("UID_MATCH_THRESH", "UID_MIN_FINGERS_PASS", "UID_MAX_SAMPLES",
                "UID_MATCH_ALGORITHM"):
        monkeypatch.delenv(var, raising=False)
    ids.init_identity_db()


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(ids.identity_bp)

    @app.route("/guarded", methods=["POST"])
    @ids.require_api_key
    def guarded():
        return "ok"
    return app.test_client()


H = {"X-API-Key": "secret"}


# ── UID format ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", ["", "SHORT", "AB12CD34EF567", "AB12CD34EF5!", "AB12 CD34EF5", None])
def test_invalid_uid_rejected(bad):
    with pytest.raises(ids.InvalidUid):
        ids.register_person(bad)


def test_uid_is_uppercased():
    ids.register_person("ab12cd34ef56")
    assert ids.enrolled_summary("Ab12Cd34Ef56")["uid"] == UID


def test_duplicate_registration_conflicts_regardless_of_case():
    ids.register_person(UID)
    with pytest.raises(ids.UidExists):
        ids.register_person(UID.lower())


# ── sample storage ───────────────────────────────────────────────────────────
def test_enrolling_unknown_uid_is_not_found():
    with pytest.raises(ids.PersonNotFound):
        ids.add_samples(UID, "single", [("2", make_template())])


def test_single_position_must_be_1_to_10():
    ids.register_person(UID)
    for bad in ("0", "11", "abc", ""):
        with pytest.raises(ids.BadRequest):
            ids.add_samples(UID, "single", [(bad, make_template())])


def test_sample_cap_is_enforced_per_finger():
    ids.register_person(UID)
    for _ in range(ids.max_samples()):
        ids.add_samples(UID, "single", [("2", make_template())])
    with pytest.raises(ids.SampleCapExceeded):
        ids.add_samples(UID, "single", [("2", make_template())])
    # a different finger is unaffected
    assert ids.add_samples(UID, "single", [("3", make_template())]) == {"3": 1}


def test_cap_is_configurable(monkeypatch):
    monkeypatch.setenv("UID_MAX_SAMPLES", "2")
    ids.register_person(UID)
    ids.add_samples(UID, "single", [("2", make_template())])
    ids.add_samples(UID, "single", [("2", make_template())])
    with pytest.raises(ids.SampleCapExceeded):
        ids.add_samples(UID, "single", [("2", make_template())])


def test_slap_enrol_is_atomic_when_one_finger_is_full(monkeypatch):
    monkeypatch.setenv("UID_MAX_SAMPLES", "1")
    ids.register_person(UID)
    ids.add_samples(UID, "slap", [("right_index", make_template(seed=1))])
    with pytest.raises(ids.SampleCapExceeded) as e:
        ids.add_samples(UID, "slap", [("right_middle", make_template(seed=2)),
                                      ("right_index", make_template(seed=3))])
    assert e.value.extra["fingers"] == ["right_index"]
    # right_middle must NOT have been written by the failed call
    assert ids.enrolled_summary(UID)["enrolled"]["slap"] == {"right_index": 1}


def test_modalities_are_separate_but_share_the_uid():
    ids.register_person(UID)
    ids.add_samples(UID, "single", [("2", make_template(seed=1))])
    ids.add_samples(UID, "slap", [("right_index", make_template(seed=2))])
    e = ids.enrolled_summary(UID)["enrolled"]
    assert e == {"single": {"2": 1}, "slap": {"right_index": 1}}


# ── single verify ────────────────────────────────────────────────────────────
def test_single_genuine_passes_and_impostor_fails():
    ids.register_person(UID)
    base = make_template(seed=1)
    ids.add_samples(UID, "single", [("2", base)])
    assert ids.verify_single(UID, 2, jitter(base))["matched"] is True
    assert ids.verify_single(UID, 2, make_template(seed=77))["matched"] is False


def test_single_matches_only_the_same_finger_position():
    ids.register_person(UID)
    base = make_template(seed=1)
    ids.add_samples(UID, "single", [("2", base)])
    # identical minutiae, but presented as a different finger: nothing enrolled there
    with pytest.raises(ids.FingerNotEnrolled):
        ids.verify_single(UID, 3, jitter(base))


def test_single_uses_best_score_over_several_samples():
    ids.register_person(UID)
    mine = make_template(seed=1)
    ids.add_samples(UID, "single", [("2", make_template(seed=50))])   # a poor sample
    ids.add_samples(UID, "single", [("2", mine)])                     # a good one
    r = ids.verify_single(UID, 2, jitter(mine))
    assert r["matched"] is True and r["samples_compared"] == 2


def test_single_does_not_see_slap_samples():
    ids.register_person(UID)
    base = make_template(seed=1)
    ids.add_samples(UID, "slap", [("2", base)])
    with pytest.raises(ids.FingerNotEnrolled):
        ids.verify_single(UID, 2, jitter(base))


def test_verify_unknown_uid_is_not_found():
    with pytest.raises(ids.PersonNotFound):
        ids.verify_single("ZZZZZZZZZZZZ", 2, make_template())


# ── slap verify ──────────────────────────────────────────────────────────────
SLAP = ["right_index", "right_middle", "right_ring", "right_little"]


def enrol_slap(n=4):
    ids.register_person(UID)
    base = {p: make_template(seed=10 + i) for i, p in enumerate(SLAP[:n])}
    ids.add_samples(UID, "slap", list(base.items()))
    return base


def probe(base, genuine):
    """genuine: set of positions presented with the real finger; others are impostors."""
    return [{"finger_position": p,
             "minutiae": jitter(m) if p in genuine else make_template(seed=900 + i)}
            for i, (p, m) in enumerate(base.items())]


def test_slap_passes_with_two_of_four():
    base = enrol_slap()
    r = ids.verify_slap(UID, probe(base, {"right_index", "right_middle"}))
    assert r["matched"] is True and r["fingers_passed"] == 2


def test_slap_fails_with_only_one_genuine_finger():
    base = enrol_slap()
    r = ids.verify_slap(UID, probe(base, {"right_index"}))
    assert r["matched"] is False and r["fingers_passed"] == 1


def test_slap_pass_rule_is_configurable(monkeypatch):
    monkeypatch.setenv("UID_MIN_FINGERS_PASS", "3")
    base = enrol_slap()
    assert ids.verify_slap(UID, probe(base, set(SLAP[:2])))["matched"] is False
    assert ids.verify_slap(UID, probe(base, set(SLAP[:3])))["matched"] is True


def test_slap_rejected_when_fewer_than_two_fingers_enrolled():
    enrol_slap(n=1)
    with pytest.raises(ids.NotEnoughEnrolledFingers) as e:
        ids.verify_slap(UID, [{"finger_position": "right_index", "minutiae": make_template()}])
    assert e.value.extra == {"enrolled_fingers": 1, "required": 2}


def test_slap_finger_without_enrolled_counterpart_is_reported_not_scored():
    base = enrol_slap(n=2)  # index + middle
    p = probe(base, set(base)) + [{"finger_position": "right_ring",
                                   "minutiae": make_template(seed=5)}]
    r = ids.verify_slap(UID, p)
    ring = next(f for f in r["fingers"] if f["finger_position"] == "right_ring")
    assert ring["enrolled"] is False and ring["score"] is None
    assert r["matched"] is True


def test_slap_uses_best_of_multiple_samples_per_finger():
    base = enrol_slap()
    ids.add_samples(UID, "slap", [(p, make_template(seed=300 + i))
                                  for i, p in enumerate(SLAP)])   # poor 2nd samples
    r = ids.verify_slap(UID, probe(base, {"right_index", "right_middle"}))
    assert r["matched"] is True


# ── API key + blueprint ──────────────────────────────────────────────────────
def test_missing_or_wrong_key_is_401(client):
    assert client.post("/guarded").status_code == 401
    assert client.post("/guarded", headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/guarded", headers=H).status_code == 200


def test_endpoints_refuse_when_server_has_no_key(client, monkeypatch):
    monkeypatch.delenv("API_KEY")
    assert client.post("/guarded", headers=H).status_code == 503


def test_register_endpoint_flow(client):
    assert client.post("/uid/register", data={"uid": UID.lower()}).status_code == 401
    r = client.post("/uid/register", data={"uid": UID.lower(), "name": "Asha"}, headers=H)
    assert r.status_code == 201 and r.get_json()["uid"] == UID
    assert client.post("/uid/register", data={"uid": UID}, headers=H).status_code == 409
    assert client.post("/uid/register", data={"uid": "short"}, headers=H).status_code == 400


def test_info_endpoint(client):
    assert client.get(f"/uid/{UID}", headers=H).status_code == 404
    ids.register_person(UID, "Asha")
    ids.add_samples(UID, "single", [("2", make_template())])
    body = client.get(f"/uid/{UID.lower()}", headers=H).get_json()
    assert body["enrolled"]["single"] == {"2": 1}
    assert body["max_samples_per_finger"] == 5
