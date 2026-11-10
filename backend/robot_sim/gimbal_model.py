"""Gimbal models from files: joint axes, offsets and boxes, nothing built in.

A model is a JSON file in user_files/gimbal_models/ (see dji_rs4.json). The
Gimbal object stores only the file name (cfg["model"]); swapping a gimbal or
correcting a measurement means picking or editing a file. Files are re-read
when they change on disk.

Format (lengths in mm, in the HANDLE's frame = the Gimbal object's mount,
with all joints at 0):

    name, about, driver           text; driver = default driver script
    joints: [ {name, sdk, axis, point, range, about}, ... ]
        in order from the handle. axis: direction (any length, need not be at
        right angles to the others); a positive angle turns right-handed
        about it. point: a point on the axis. sdk: which value of the gimbal
        reading / frames it is ("yaw", "roll" or "pitch" for DJI R SDK
        joints J1-J3).
    aim_joints: [name, name]      joints Look At may turn (others held at 0)
    camera: {origin, lens, right}
        camera centre, lens and right directions with all joints at 0. No
        camera body is drawn: the Gimbal's Camera object is parented here.
    parts: [ ... ]                what to draw, each on a link:
        link 0 = the handle (drawn once, grey), link k = moves with joint k
        {from, to, width}        a bar between two points
        {center, size}           a box, sides along x / y / z
        {center, radius}         a sphere

Kinematics: product of exponentials. Joint k with unit axis w and point p
turns by angle a: R = rot(w, a), translation p - R p, applied in the frame
of the joint before it. The camera frame (z = lens, x = right, y = image
down) sits at camera.origin on the last joint.
"""
import json
import math
import os

import numpy as np

import app_paths

SDK_KEYS = ("yaw", "roll", "pitch")
DEFAULT_FILE = "dji_rs4.json"
_cache = {}  # path -> (mtime, model or None, error)


# ---------------------------------------------------------------------------
# Small maths (wxyz quaternions)
# ---------------------------------------------------------------------------
def qmul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])


def qrot(q, v):
    w, x, y, z = q
    v = np.asarray(v, dtype=float)
    uv = 2.0 * np.cross([x, y, z], v)
    return v + w * uv + np.cross([x, y, z], uv)


def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qaxis(axis, deg):
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) or 1.0)
    h = math.radians(float(deg)) * 0.5
    return np.concatenate([[math.cos(h)], a * math.sin(h)])


def qfrom_matrix(m):
    m = np.asarray(m, dtype=float)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / (np.linalg.norm(q) or 1.0)


def qbetween(a, b):
    """Rotation turning direction a onto direction b."""
    a = np.asarray(a, float) / (np.linalg.norm(a) or 1.0)
    b = np.asarray(b, float) / (np.linalg.norm(b) or 1.0)
    c = float(np.dot(a, b))
    if c < -0.999999:
        perp = np.cross(a, [1, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0, 1, 0])
        return qaxis(perp, 180.0)
    v = np.cross(a, b)
    q = np.array([1.0 + c, v[0], v[1], v[2]])
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def models_dir():
    return app_paths.GIMBAL_MODELS_DIR


def list_models():
    d = models_dir()
    try:
        return sorted(f for f in os.listdir(d) if f.lower().endswith(".json"))
    except OSError:
        return []


def path_of(name):
    name = str(name or "").strip() or DEFAULT_FILE
    return name if os.path.isabs(name) else os.path.join(str(models_dir()), name)


def _vec(v, n=3, scale=1.0, what="value"):
    a = [float(x) for x in list(v)]
    if len(a) != n:
        raise ValueError("%s needs %d numbers" % (what, n))
    return np.asarray(a) * scale


def parse(raw):
    """Model dict (metres, unit axes) from a file's JSON. Raises ValueError."""
    mm = 0.001
    joints = []
    for i, j in enumerate(raw.get("joints") or []):
        axis = _vec(j.get("axis"), what="joint %d axis" % (i + 1))
        if np.linalg.norm(axis) < 1e-9:
            raise ValueError("joint %d axis is zero" % (i + 1))
        sdk = str(j.get("sdk") or SDK_KEYS[min(i, 2)])
        if sdk not in SDK_KEYS:
            raise ValueError("joint %d sdk must be one of %s" % (i + 1, ", ".join(SDK_KEYS)))
        rng = j.get("range") or [-180, 180]
        joints.append({"name": str(j.get("name") or "J%d" % (i + 1)), "sdk": sdk,
                       "axis": axis / np.linalg.norm(axis),
                       "point": _vec(j.get("point") or [0, 0, 0], scale=mm, what="joint %d point" % (i + 1)),
                       "range": [float(rng[0]), float(rng[1])]})
    if not joints:
        raise ValueError("no joints")
    cam = raw.get("camera") or {}
    lens = _vec(cam.get("lens") or [1, 0, 0], what="camera lens")
    right = _vec(cam.get("right") or [0, -1, 0], what="camera right")
    lens = lens / (np.linalg.norm(lens) or 1.0)
    right = right - lens * float(np.dot(right, lens))
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("camera right is along the lens")
    right = right / np.linalg.norm(right)
    down = np.cross(lens, right)
    cam_q = qfrom_matrix(np.column_stack([right, down, lens]))  # camera x, y, z
    names = [j["name"] for j in joints]
    aim = [n for n in (raw.get("aim_joints") or [names[0], names[-1]]) if n in names][:2]
    parts = []
    for p in raw.get("parts") or []:
        link = int(p.get("link", 0))
        if not 0 <= link <= len(joints):
            raise ValueError("part %r: link %d out of range" % (p.get("name"), link))
        rec = {"link": link, "name": str(p.get("name") or "part")}
        if "from" in p and "to" in p:
            rec.update(kind="bar", a=_vec(p["from"], scale=mm), b=_vec(p["to"], scale=mm),
                       width=float(p.get("width") or 20) * mm)
        elif "radius" in p:
            rec.update(kind="sphere", center=_vec(p.get("center") or [0, 0, 0], scale=mm),
                       radius=float(p["radius"]) * mm)
        else:
            rec.update(kind="box", center=_vec(p.get("center") or [0, 0, 0], scale=mm),
                       size=_vec(p.get("size") or [20, 20, 20], scale=mm))
        parts.append(rec)
    return {
        "name": str(raw.get("name") or "gimbal"),
        "driver": str(raw.get("driver") or ""),
        "joints": joints,
        "aim": [names.index(n) for n in aim],
        "cam_origin": _vec(cam.get("origin") or [0, 0, 0], scale=mm, what="camera origin"),
        "cam_q": cam_q,
        "parts": parts,
    }


# A gimbal with no usable model file: three plain axes through the mount
# origin, no parts, so the sandbox still shows something and says why.
_FALLBACK = parse({"joints": [{"sdk": "yaw", "axis": [0, 0, -1]}, {"sdk": "roll", "axis": [1, 0, 0]},
                              {"sdk": "pitch", "axis": [0, -1, 0]}],
                   "camera": {"lens": [1, 0, 0], "right": [0, -1, 0]}})
_FALLBACK["name"] = "no model"


def load(name):
    """(model, error, mtime). Re-reads the file when it changes; on any
    problem returns the fallback model and the error text."""
    path = path_of(name)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return _FALLBACK, "model file not found: %s" % path, None
    hit = _cache.get(path)
    if hit and hit[0] == mtime:
        return (hit[1] or _FALLBACK), hit[2], mtime
    try:
        with open(path, "r", encoding="utf-8") as f:
            model = parse(json.load(f))
        err = ""
    except Exception as e:
        model, err = None, "model %s: %s" % (os.path.basename(path), e)
        print("gimbal", err)
    _cache[path] = (mtime, model, err)
    return (model or _FALLBACK), err, mtime


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def joint_values(model, target):
    """Joint angles in model order from a stored [yaw, roll, pitch] list."""
    t = (list(target or [0, 0, 0]) + [0, 0, 0])[:3]
    return [float(t[SDK_KEYS.index(j["sdk"])]) for j in model["joints"]]


def joint_local(model, k, deg):
    """(wxyz, position) of joint k's node in the frame of the one before it."""
    j = model["joints"][k]
    q = qaxis(j["axis"], deg)
    return q, j["point"] - qrot(q, j["point"])


def camera_pose(model, angles):
    """Camera (wxyz, position) in the handle frame for joint angles (model order)."""
    q = np.array([1.0, 0, 0, 0])
    p = np.zeros(3)
    for k, a in enumerate(angles):
        lq, lp = joint_local(model, k, a)
        p = p + qrot(q, lp)
        q = qmul(q, lq)
    return qmul(q, model["cam_q"]), p + qrot(q, model["cam_origin"])


def aim(model, mount_pos, mount_wxyz, world_target, prev):
    """Stored [yaw, roll, pitch] that point the lens at world_target, turning
    only the model's aim joints (from prev; the others at 0). None if the
    target is at the camera."""
    from scipy.optimize import minimize
    tgt = qrot(qinv(np.asarray(mount_wxyz, float)), np.asarray(world_target, float) - np.asarray(mount_pos, float))
    base = joint_values(model, prev)
    ai = model["aim"]
    angles0 = [0.0] * len(model["joints"])

    def cost(x):
        a = list(angles0)
        for i, k in enumerate(ai):
            a[k] = x[i]
        q, p = camera_pose(model, a)
        d = tgt - p
        n = np.linalg.norm(d)
        if n < 1e-6:
            return 0.0
        return 1.0 - float(np.dot(qrot(q, [0, 0, 1]), d / n))

    _, p0 = camera_pose(model, angles0)
    if np.linalg.norm(tgt - p0) < 1e-6:
        return None
    starts = [[base[k] for k in ai]] + [[a, b] for a in (-135, -45, 45, 135) for b in (-60, 0, 60)]
    best = None
    for s in starts:
        r = minimize(cost, np.asarray(s[:len(ai)], float), method="Nelder-Mead",
                     options={"xatol": 0.01, "fatol": 1e-9, "maxiter": 400})
        # Prefer the solution nearest the previous angles among equally good ones.
        dist = sum(abs(((r.x[i] - base[k]) + 180) % 360 - 180) for i, k in enumerate(ai))
        key = (round(float(r.fun), 6), dist)
        if best is None or key < best[0]:
            best = (key, r.x)
        if r.fun < 1e-8 and s is starts[0]:
            break
    a = list(angles0)
    for i, k in enumerate(ai):
        a[k] = ((float(best[1][i]) + 180) % 360) - 180
    out = [0.0, 0.0, 0.0]
    for k, j in enumerate(model["joints"]):
        out[SDK_KEYS.index(j["sdk"])] = a[k]
    return out


def link_poses(model, angles):
    """[(wxyz, position)] of link 0 (handle) .. link n in the handle frame."""
    q = np.array([1.0, 0, 0, 0])
    p = np.zeros(3)
    out = [(q, p)]
    for k in range(len(model["joints"])):
        lq, lp = joint_local(model, k, angles[k] if k < len(angles) else 0.0)
        p = p + qrot(q, lp)
        q = qmul(q, lq)
        out.append((q, p))
    return out


def world_parts(model, angles, mount_pos, mount_wxyz, margin=0.0, min_link=1):
    """The model's parts on links >= min_link, in the world, for colliders:
    [{"kind": "box", "c", "q", "h"} | {"kind": "sphere", "c", "r"}], grown by margin."""
    mq = np.asarray(mount_wxyz, dtype=float)
    mq = mq / (np.linalg.norm(mq) or 1.0)
    mp = np.asarray(mount_pos, dtype=float)
    links = link_poses(model, angles)
    out = []
    for part in model["parts"]:
        if part["link"] < min_link:
            continue
        lq, lp = links[part["link"]]
        wq = qmul(mq, lq)
        to_world = lambda v: mp + qrot(mq, lp + qrot(lq, v))
        if part["kind"] == "sphere":
            out.append({"kind": "sphere", "c": to_world(part["center"]), "r": part["radius"] + margin})
        elif part["kind"] == "bar":
            a, b = part["a"], part["b"]
            d = b - a
            L = float(np.linalg.norm(d)) or 1e-4
            out.append({"kind": "box", "c": to_world((a + b) / 2.0), "q": qmul(wq, qbetween([1, 0, 0], d)),
                        "h": np.array([L, part["width"], part["width"]]) / 2.0 + margin})
        else:
            out.append({"kind": "box", "c": to_world(part["center"]), "q": wq,
                        "h": np.asarray(part["size"], dtype=float) / 2.0 + margin})
    return out


# ---------------------------------------------------------------------------
# World orientation <-> joints
# ---------------------------------------------------------------------------
# A camera's world orientation as yaw (heading of the lens about world up),
# pitch (lens above the horizon, positive up) and roll (picture turned, positive
# = right side down), in degrees. Camera frame: z = lens, x = right, y = image down.
def world_ypr(cam_wxyz):
    q = np.asarray(cam_wxyz, dtype=float)
    f, r = qrot(q, [0, 0, 1]), qrot(q, [1, 0, 0])
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
    yaw = math.degrees(math.atan2(f[1], f[0])) if math.hypot(f[0], f[1]) > 1e-6 else 0.0
    y = math.radians(yaw)
    r0 = np.array([math.sin(y), -math.cos(y), 0.0])     # level right of the lens
    u0 = np.cross(r0, f)                                  # up, square to the lens
    roll = math.degrees(math.atan2(-float(np.dot(r, u0)), float(np.dot(r, r0))))
    return [yaw, pitch, roll]


def world_q(yaw, pitch, roll):
    """Camera world wxyz for yaw / pitch / roll (world_ypr's inverse)."""
    y, p, r = (math.radians(float(v)) for v in (yaw, pitch, roll))
    f = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
    r0 = np.array([math.sin(y), -math.cos(y), 0.0])
    u0 = np.cross(r0, f)
    right = r0 * math.cos(r) - u0 * math.sin(r)
    down = np.cross(f, right)
    return qfrom_matrix(np.column_stack([right, down, f]))


def _qangle(a, b):
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return math.degrees(2.0 * math.acos(min(1.0, d)))


def ik(model, mount_wxyz, cam_world_wxyz, prev):
    """Joint angles (stored [yaw, roll, pitch] order) that turn the camera to
    cam_world_wxyz with the handle at mount_wxyz, nearest the previous angles.
    Returns (angles, error in degrees: 0 when reachable)."""
    from scipy.optimize import least_squares
    mq = np.asarray(mount_wxyz, float)
    mq = mq / (np.linalg.norm(mq) or 1.0)
    want = qmul(qinv(mq), np.asarray(cam_world_wxyz, float))  # in the handle frame
    want = want / (np.linalg.norm(want) or 1.0)
    base = joint_values(model, prev)

    def resid(x):
        q, _ = camera_pose(model, list(x))
        d = qmul(qinv(q), want)
        if d[0] < 0:
            d = -d
        v = d[1:]
        s = float(np.linalg.norm(v))
        ang = 2.0 * math.atan2(s, d[0])
        rv = v / s * ang if s > 1e-12 else v * 2.0
        # Small pull towards the previous angles, so equal solutions keep the nearest.
        return np.concatenate([np.degrees(rv), 0.002 * (np.asarray(x) - base)])

    starts = [base]
    for k in range(len(base)):
        for dv in (180.0, -180.0, 90.0, -90.0):
            s = list(base)
            s[k] += dv
            starts.append(s)
    best = None
    for s in starts:
        r = least_squares(resid, np.asarray(s, float), method="lm", xtol=1e-10, ftol=1e-10, max_nfev=400)
        q, _ = camera_pose(model, list(r.x))
        err = _qangle(q, want)
        dist = sum(abs(((a - b) + 180) % 360 - 180) for a, b in zip(r.x, base))
        key = (round(err, 2), dist)
        if best is None or key < best[0]:
            best = (key, r.x, err)
        if err < 0.01 and s is starts[0]:
            break
    angles = [((float(a) + 180) % 360) - 180 for a in best[1]]
    out = [0.0, 0.0, 0.0]
    for k, j in enumerate(model["joints"]):
        out[SDK_KEYS.index(j["sdk"])] = angles[k]
    return out, best[2]


# ---------------------------------------------------------------------------
# Calibration: J2's axis from one reading in a known pose
# ---------------------------------------------------------------------------
CAL_MIN_J2 = 30.0  # J2 must have turned this far, or its axis makes no difference


def _cam_pitch_roll(model, angles):
    """Camera pitch / roll (deg) with the handle unrotated and its hand end up
    (handle frame z = world up): pitch = lens elevation, roll = image tilt."""
    q, _ = camera_pose(model, angles)
    lens, down = qrot(q, [0, 0, 1]), qrot(q, [0, 1, 0])
    right = np.cross(down, lens)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, lens[2]))))
    roll = math.degrees(math.asin(max(-1.0, min(1.0, -right[2]))))
    return pitch, roll, lens, down


CAL_MAX_ERR = 3.0  # deg: the best fit must put the camera within this of the measured pitch / roll


def fit_j2_axis(raw, reading, world_pitch, world_roll):
    """Fit J2's axis in a model file's JSON (raw) to one reading taken with the
    gimbal held HAND END UP (handle vertical) and the gimbal's own world pitch /
    roll. Which way the camera faces around the handle doesn't matter. J2's
    axis is the lens line at zero tilted towards the hand end ("tilt") and
    sideways ("side"); among fits that match, the one nearest the model's
    current axis wins. Returns {axis, tilt, side, pitch, roll, error} or
    raises ValueError with the reading and the best fit in the message."""
    from scipy.optimize import minimize
    import copy
    names = [j.get("sdk") for j in raw.get("joints") or []]
    if "roll" not in names:
        raise ValueError("the model has no J2 (sdk 'roll') joint")
    k2 = names.index("roll")
    reading = [float(v) for v in (list(reading) + [0, 0, 0])[:3]]
    seen = "reading J1 %.1f, J2 %.1f, J3 %.1f; world pitch %.1f, roll %.1f" % (
        reading[0], reading[1], reading[2], world_pitch, world_roll)
    if abs(((reading[1] + 180) % 360) - 180) < CAL_MIN_J2:
        raise ValueError("J2 is near 0, where its axis makes no difference, so this pose can't calibrate it "
                         "(in the reference pose the RS 4 reads J2 near 180). " + seen)
    lens0 = np.asarray((raw.get("camera") or {}).get("lens") or [1, 0, 0], float)
    lens0 /= np.linalg.norm(lens0)
    up0 = np.array([0.0, 0.0, 1.0])

    def axis_of(tilt, side):
        a = qrot(qaxis(up0, side), lens0)                    # sideways about the handle axis
        perp = np.cross(a, up0)
        return qrot(qaxis(perp, tilt), a) if np.linalg.norm(perp) > 1e-9 else a  # towards the hand end

    def model_for(tilt, side):
        r = copy.deepcopy(raw)
        r["joints"][k2]["axis"] = [float(v) for v in axis_of(tilt, side)]
        return parse(r)

    def errors(x):
        m = model_for(x[0], x[1])
        p, ro, lens, down = _cam_pitch_roll(m, joint_values(m, reading))
        return p - world_pitch, ro - world_roll, float(down[2])

    def cost(x):
        dp, dr, dz = errors(x)
        # The image the right way up (image-down pointing down).
        return dp * dp + 0.3 * dr * dr + 400.0 * max(0.0, dz)

    # Current tilt / side of the model's J2 axis, to prefer the nearest fit.
    cur = np.asarray(raw["joints"][k2].get("axis") or lens0, float)
    cur /= (np.linalg.norm(cur) or 1.0)
    cur_tilt = math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(cur, up0))))))
    flat = cur - up0 * float(np.dot(cur, up0))
    cur_side = math.degrees(math.atan2(float(np.cross(lens0, flat)[2]), float(np.dot(lens0, flat)))) if np.linalg.norm(flat) > 1e-6 else 0.0
    starts = [(cur_tilt, cur_side)] + [(t, sd) for t in range(-85, 86, 10) for sd in (-20, 0, 20)]
    fits = []
    for st in starts:
        r = minimize(cost, np.asarray(st, float), method="Nelder-Mead", options={"xatol": 0.01, "fatol": 1e-6, "maxiter": 400})
        dp, dr, dz = errors(r.x)
        err = max(abs(dp), abs(dr))
        fits.append((err if dz < 0 else 999.0, abs(r.x[0] - cur_tilt) + abs(r.x[1] - cur_side), r.x))
    good = [f for f in fits if f[0] <= CAL_MAX_ERR]
    best = min(good, key=lambda f: f[1]) if good else min(fits, key=lambda f: f[0])
    tilt, side = float(best[2][0]), float(best[2][1])
    m = model_for(tilt, side)
    p, ro, lens, down = _cam_pitch_roll(m, joint_values(m, reading))
    if not good:
        raise ValueError("no J2 tilt makes this reading match (best: tilt %.1f, side %.1f puts the camera at "
                         "pitch %.1f, roll %.1f%s). Is the handle vertical, hand end up? %s" % (
                             tilt, side, p, ro, ", upside down" if down[2] > 0 else "", seen))
    return {"axis": [round(float(v), 4) for v in axis_of(tilt, side)], "tilt": round(tilt, 2), "side": round(side, 2),
            "was_tilt": round(cur_tilt, 2), "was_side": round(cur_side, 2),
            "pitch": round(p, 2), "roll": round(ro, 2), "error": round(float(best[0]), 2), "reading": seen}


def write_axis(name, sdk, axis, note):
    """Set one joint's axis in a model file (a .bak copy is kept first)."""
    import shutil
    path = path_of(name)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for j in raw.get("joints") or []:
        if j.get("sdk") == sdk:
            j["axis"] = [float(v) for v in axis]
            j["calibrated"] = note
            break
    else:
        raise ValueError("no %s joint in %s" % (sdk, os.path.basename(path)))
    shutil.copyfile(path, path + ".bak")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Scene (viser)
# ---------------------------------------------------------------------------
def _add_part(server, name, part, rgb, opacity, on_click, out=None):
    if part["kind"] == "bar":
        a, b = part["a"], part["b"]
        d = b - a
        L = float(np.linalg.norm(d)) or 1e-4
        h = server.scene.add_box(name=name, dimensions=(L, part["width"], part["width"]), color=rgb,
                                 wxyz=tuple(qbetween([1, 0, 0], d)), position=tuple((a + b) / 2.0))
    elif part["kind"] == "sphere":
        h = server.scene.add_icosphere(name=name, radius=part["radius"], color=rgb, position=tuple(part["center"]))
    else:
        h = server.scene.add_box(name=name, dimensions=tuple(part["size"]), color=rgb, position=tuple(part["center"]))
    try:
        h.opacity = opacity
    except Exception:
        pass
    if on_click is not None:
        try:
            h.on_click(on_click)
        except Exception:
            pass
    if out is not None:
        out.append(h)
    return h


def build_base(server, name, model, rgb, on_click):
    """Link-0 parts (the handle) under `name`; returns the root node."""
    root = server.scene.add_frame(name, show_axes=False)
    for i, part in enumerate(model["parts"]):
        if part["link"] == 0:
            _add_part(server, "%s/p%d" % (name, i), part, rgb, 0.7, on_click)
    return root


def build_twin(server, name, model, rgb, angles, on_click, arrow_len=0.24):
    """One twin: nested joint nodes with their link parts, and a camera frame
    with lens / right arrows (no body). Returns (root, [joint nodes], [part handles])."""
    root = server.scene.add_frame(name, show_axes=False)
    nodes = []
    parts = []
    path = name
    for k in range(len(model["joints"])):
        path = "%s/j%d" % (path, k + 1)
        q, p = joint_local(model, k, angles[k] if k < len(angles) else 0.0)
        nodes.append(server.scene.add_frame(path, show_axes=False, wxyz=tuple(q), position=tuple(p)))
        for i, part in enumerate(model["parts"]):
            if part["link"] == k + 1:
                _add_part(server, "%s/p%d" % (path, i), part, rgb, 0.55, on_click, parts)
    cam = path + "/camera"
    server.scene.add_frame(cam, show_axes=False, wxyz=tuple(model["cam_q"]), position=tuple(model["cam_origin"]))
    for sub, axis, length, shaft, head in (("lens", 2, arrow_len, 0.007, 0.018), ("side", 0, arrow_len * 0.4, 0.005, 0.012)):
        pts = np.zeros((1, 2, 3), dtype=np.float32)
        pts[0, 1, axis] = float(length)
        server.scene.add_arrows(name="%s/%s" % (cam, sub), points=pts, colors=np.asarray(rgb, dtype=np.uint8).reshape(3),
                                shaft_radius=shaft, head_radius=head, head_length=head * 2.2)
    return root, nodes, parts


def pose_twin(model, nodes, angles):
    for k, node in enumerate(nodes):
        q, p = joint_local(model, k, angles[k] if k < len(angles) else 0.0)
        try:
            node.wxyz = tuple(float(v) for v in q)
            node.position = tuple(float(v) for v in p)
        except Exception:
            pass
