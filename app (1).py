"""
Face Match ML — Single-file Flask App
======================================
Web app untuk training model PCA/SVD dari dataset wajah (sintetis atau upload ZIP),
lalu membandingkan dua foto untuk menentukan apakah orang yang sama.

Cara jalankan:
    pip install flask opencv-python numpy scikit-learn
    python app.py
    buka http://127.0.0.1:5000

Format ZIP dataset:
    Semua foto rata (tanpa folder), nama file = "namaorang_nomor.ext"
    Contoh: andi_1.jpg, andi_2.jpg, budi_1.jpg, budi_2.png, siti_1.jpeg ...
    Nama orang diambil dari bagian sebelum underscore + angka terakhir.
"""

from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import base64, os, re, pickle, threading, zipfile, tempfile, shutil

app = Flask(__name__)

# ── Konfigurasi ──────────────────────────────
IMG_SIZE        = (100, 100)
N_COMPONENTS    = 50
MODEL_PATH      = "model.pkl"
UPLOAD_TMP_DIR  = "uploads_tmp"
ALLOWED_EXT     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

os.makedirs(UPLOAD_TMP_DIR, exist_ok=True)

# ── State global (in-memory) ─────────────────
model_store = {"pca": None, "X_pca": None, "labels": None, "trained": False}
train_log   = []


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_image(img: np.ndarray, use_face_detection=True) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_face_detection:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces   = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            gray = gray[y:y + h, x:x + w]
    return cv2.resize(gray, IMG_SIZE).astype(np.float64).flatten() / 255.0


def get_face_thumbnail(img: np.ndarray) -> str:
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces   = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    crop    = img
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        pad = int(min(w, h) * 0.15)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        crop = img[y1:y2, x1:x2]
    crop_r = cv2.resize(crop, (150, 150))
    _, buf = cv2.imencode(".jpg", crop_r, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def decode_b64(s: str) -> np.ndarray:
    if "," in s:
        s = s.split(",")[1]
    arr = np.frombuffer(base64.b64decode(s), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak dapat dibaca")
    return img


def extract_label_from_filename(filename: str) -> str:
    """
    Ambil nama orang dari nama file format: namaorang_nomor.ext
    Contoh: 'andi_1.jpg' -> 'andi' | 'siti_wulan_03.png' -> 'siti_wulan'
    """
    stem = os.path.splitext(filename)[0]
    match = re.match(r"^(.*)_\d+$", stem)
    if match:
        return match.group(1).strip().lower()
    return stem.strip().lower()


# ─────────────────────────────────────────────
# DATASET SINTETIS
# ─────────────────────────────────────────────

def create_synthetic_dataset(n_persons=5, n_images=8):
    """Buat dataset sintetis: setiap orang punya 'wajah dasar' unik + variasi noise."""
    train_log.append("Membuat dataset sintetis...")
    X, labels = [], []
    names = [f"orang_{i+1}" for i in range(n_persons)]

    for name in names:
        rng       = np.random.default_rng(abs(hash(name)) % 2**31)
        base_face = rng.integers(40, 220, IMG_SIZE).astype(np.float64)

        cx, cy = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2
        for px in range(IMG_SIZE[0]):
            for py in range(IMG_SIZE[1]):
                dist = ((px - cx) / 40) ** 2 + ((py - cy) / 50) ** 2
                if dist < 1.0:
                    base_face[px, py] *= 0.75

        for j in range(n_images):
            noise = rng.integers(-25, 26, IMG_SIZE).astype(np.float64)
            face  = np.clip(base_face + noise, 0, 255) / 255.0
            X.append(face.flatten())
            labels.append(name)

        train_log.append(f"  v {name}: {n_images} gambar dibuat")

    return np.array(X), np.array(labels)


# ─────────────────────────────────────────────
# DATASET DARI ZIP UPLOAD
# ─────────────────────────────────────────────

def load_dataset_from_zip(zip_path: str):
    """
    Ekstrak ZIP, baca semua gambar (flat scan ke semua subfolder),
    label diambil dari nama file pakai extract_label_from_filename().
    """
    train_log.append("Mengekstrak dan membaca dataset dari ZIP...")

    extract_dir = tempfile.mkdtemp(prefix="faceds_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        X, labels = [], []
        skipped = 0

        for root, _, files in os.walk(extract_dir):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ALLOWED_EXT:
                    continue
                if fname.startswith("._") or fname.startswith("."):
                    continue

                fpath = os.path.join(root, fname)
                img = cv2.imread(fpath)
                if img is None:
                    skipped += 1
                    continue

                try:
                    vec = preprocess_image(img, use_face_detection=True)
                except Exception:
                    skipped += 1
                    continue

                label = extract_label_from_filename(fname)
                X.append(vec)
                labels.append(label)

        if len(X) == 0:
            raise ValueError("Tidak ada gambar valid ditemukan di dalam ZIP")

        unique_labels = sorted(set(labels))
        train_log.append(f"  v {len(X)} gambar berhasil dibaca, {len(unique_labels)} orang terdeteksi")
        if skipped > 0:
            train_log.append(f"  ! {skipped} file dilewati (gagal dibaca / format tidak didukung)")
        for lbl in unique_labels:
            count = labels.count(lbl)
            train_log.append(f"    - {lbl}: {count} foto")

        return np.array(X), np.array(labels)

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# TRAINING PCA
# ─────────────────────────────────────────────

def run_training(mode: str, n_persons=5, n_images=8, zip_path=None):
    global model_store, train_log
    train_log = ["Memulai training..."]

    try:
        if mode == "zip":
            X, labels = load_dataset_from_zip(zip_path)
        else:
            X, labels = create_synthetic_dataset(n_persons, n_images)

        n_unique = len(np.unique(labels))
        if n_unique < 2:
            raise ValueError("Minimal perlu 2 orang berbeda dalam dataset untuk training")
        if X.shape[0] < 3:
            raise ValueError("Minimal perlu 3 foto total dalam dataset untuk training")

        train_log.append(f"\nDataset: {X.shape[0]} gambar | {n_unique} orang | dimensi {X.shape[1]}")

        n_comp = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1])
        train_log.append(f"\nTraining PCA dengan {n_comp} komponen...")
        pca   = PCA(n_components=n_comp, svd_solver="full")
        X_pca = pca.fit_transform(X)

        explained = float(np.sum(pca.explained_variance_ratio_)) * 100
        train_log.append(f"  v Explained variance: {explained:.1f}%")
        train_log.append(f"  v Dimensi PCA: {X_pca.shape}")

        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"pca": pca, "X_pca": X_pca, "labels": labels}, f)

        model_store["pca"]     = pca
        model_store["X_pca"]   = X_pca
        model_store["labels"]  = labels
        model_store["trained"] = True

        train_log.append(f"\nModel berhasil ditraining! Siap membandingkan wajah.")

    except Exception as e:
        train_log.append(f"\nError: {str(e)}")
    finally:
        if zip_path and os.path.exists(zip_path):
            os.remove(zip_path)


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    if not model_store["trained"] and os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            model_store.update({"pca": data["pca"], "X_pca": data["X_pca"],
                                 "labels": data["labels"], "trained": True})
        except Exception:
            pass
    return render_template_string(INDEX_HTML)


@app.route("/train_synthetic", methods=["POST"])
def train_synthetic():
    body      = request.get_json() or {}
    n_persons = max(2, min(int(body.get("n_persons", 5)), 20))
    n_images  = max(3, min(int(body.get("n_images", 8)), 20))

    model_store["trained"] = False
    t = threading.Thread(target=run_training, kwargs={
        "mode": "synthetic", "n_persons": n_persons, "n_images": n_images
    }, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/train_zip", methods=["POST"])
def train_zip():
    if "dataset" not in request.files:
        return jsonify({"error": "File ZIP tidak ditemukan"}), 400

    file = request.files["dataset"]
    if file.filename == "" or not file.filename.lower().endswith(".zip"):
        return jsonify({"error": "Harap upload file berformat .zip"}), 400

    save_path = os.path.join(UPLOAD_TMP_DIR, "dataset_upload.zip")
    file.save(save_path)

    model_store["trained"] = False
    t = threading.Thread(target=run_training, kwargs={
        "mode": "zip", "zip_path": save_path
    }, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/train_status")
def train_status():
    return jsonify({"trained": model_store["trained"], "log": "\n".join(train_log)})


@app.route("/compare", methods=["POST"])
def compare():
    if not model_store["trained"]:
        return jsonify({"error": "Model belum ditraining. Silakan train dulu!"}), 400

    try:
        data = request.get_json()
        img1 = decode_b64(data["image1"])
        img2 = decode_b64(data["image2"])

        thumb1 = get_face_thumbnail(img1)
        thumb2 = get_face_thumbnail(img2)

        vec1 = preprocess_image(img1)
        vec2 = preprocess_image(img2)

        pca = model_store["pca"]
        z1  = pca.transform(vec1.reshape(1, -1))
        z2  = pca.transform(vec2.reshape(1, -1))

        cos_sim  = float(cosine_similarity(z1, z2)[0][0])
        euc_dist = float(euclidean_distances(z1, z2)[0][0])
        euc_sim  = max(0, 1.0 - min(euc_dist / 30.0, 1.0))
        combined = cos_sim * 0.6 + euc_sim * 0.4

        if combined >= 0.85:
            verdict, verdict_sub, level = "SANGAT MIRIP", "Kemungkinan besar orang yang sama", "high"
        elif combined >= 0.70:
            verdict, verdict_sub, level = "CUKUP MIRIP", "Mungkin orang yang sama", "medium"
        elif combined >= 0.50:
            verdict, verdict_sub, level = "KURANG MIRIP", "Kemungkinan orang berbeda", "low"
        else:
            verdict, verdict_sub, level = "TIDAK MIRIP", "Kemungkinan besar orang berbeda", "none"

        return jsonify({
            "success": True,
            "cosine_similarity": round(cos_sim * 100, 1),
            "euclidean_sim": round(euc_sim * 100, 1),
            "combined_score": round(combined * 100, 1),
            "verdict": verdict,
            "verdict_sub": verdict_sub,
            "verdict_level": level,
            "thumbnail1": thumb1,
            "thumbnail2": thumb2,
            "n_train_samples": len(model_store["labels"]),
            "n_persons": len(np.unique(model_store["labels"])),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model_info")
def model_info():
    if not model_store["trained"]:
        return jsonify({"trained": False})
    labels  = model_store["labels"]
    persons = np.unique(labels).tolist()
    return jsonify({
        "trained": True,
        "n_samples": len(labels),
        "n_persons": len(persons),
        "persons": persons,
        "n_components": int(model_store["pca"].n_components_),
        "explained_variance": round(float(np.sum(model_store["pca"].explained_variance_ratio_)) * 100, 1),
    })


# ─────────────────────────────────────────────
# HTML TEMPLATE (embedded, single-file)
# ─────────────────────────────────────────────

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Face Match — PCA Face Recognition</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
:root{
  --bg:#0f1117;--surface:#1a1d27;--surface2:#222535;--border:#2e3348;
  --accent:#6c63ff;--accent2:#9f97ff;--text:#e8eaf6;--muted:#7b80a0;
  --green:#4caf87;--yellow:#f5c842;--orange:#f58a42;--red:#f55a5a;--radius:16px;
}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
     min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:28px 16px 64px;}
header{text-align:center;margin-bottom:28px;}
.logo{display:inline-flex;align-items:center;gap:12px;margin-bottom:10px;}
.logo-icon{width:46px;height:46px;background:linear-gradient(135deg,var(--accent),var(--accent2));
           border-radius:13px;display:flex;align-items:center;justify-content:center;font-size:22px;}
h1{font-size:1.9rem;font-weight:700;background:linear-gradient(135deg,var(--accent2),#fff);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.subtitle{color:var(--muted);font-size:.9rem;margin-top:5px;}
.tabs{display:flex;gap:8px;margin-bottom:20px;width:100%;max-width:760px;}
.tab{flex:1;padding:10px;border:1px solid var(--border);border-radius:10px;background:var(--surface);
     color:var(--muted);font-size:.85rem;font-weight:600;cursor:pointer;text-align:center;transition:all .2s;}
.tab:hover{border-color:var(--accent);color:var(--text);}
.tab.active{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;border-color:transparent;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
      padding:28px;width:100%;max-width:760px;box-shadow:0 8px 40px rgba(0,0,0,.4);}
.section{display:none;}
.section.active{display:block;}

/* Sub-tabs untuk metode training */
.subtabs{display:flex;gap:8px;margin-bottom:20px;}
.subtab{flex:1;padding:9px;border:1px solid var(--border);border-radius:8px;background:var(--surface2);
        color:var(--muted);font-size:.8rem;font-weight:600;cursor:pointer;text-align:center;transition:all .2s;}
.subtab:hover{border-color:var(--accent);}
.subtab.active{background:rgba(108,99,255,.15);border-color:var(--accent);color:var(--accent2);}
.subsection{display:none;}
.subsection.active{display:block;}

.train-config{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;}
.field label{display:block;font-size:.8rem;color:var(--muted);margin-bottom:6px;}
.field input[type=range]{width:100%;accent-color:var(--accent);}
.field .val{font-size:1.1rem;font-weight:700;color:var(--accent2);}

.dropzone{border:2px dashed var(--border);border-radius:12px;background:var(--surface2);
          padding:32px 20px;text-align:center;cursor:pointer;transition:all .2s;margin-bottom:16px;}
.dropzone:hover,.dropzone.drag-over{border-color:var(--accent);background:rgba(108,99,255,.05);}
.dropzone.has-file{border-color:var(--green);border-style:solid;}
.dropzone-icon{font-size:2.2rem;margin-bottom:8px;}
.dropzone-label{font-size:.88rem;color:var(--muted);}
.dropzone-sub{font-size:.74rem;color:var(--border);margin-top:4px;}
.dropzone input[type=file]{display:none;}
.file-chosen{color:var(--green);font-weight:600;font-size:.85rem;margin-top:6px;}

.format-hint{font-size:.78rem;color:var(--muted);background:var(--surface2);border:1px solid var(--border);
             border-radius:8px;padding:12px 14px;margin-bottom:16px;line-height:1.6;}
.format-hint code{background:rgba(108,99,255,.15);color:var(--accent2);padding:1px 6px;border-radius:4px;}

.log-box{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:16px;
         font-family:monospace;font-size:.78rem;color:#a8b0d0;min-height:120px;max-height:240px;
         overflow-y:auto;white-space:pre-wrap;line-height:1.6;display:none;margin-bottom:16px;}

.model-badge{display:none;background:rgba(76,175,135,.12);border:1px solid rgba(76,175,135,.3);
            border-radius:10px;padding:14px 18px;margin-bottom:18px;}
.model-badge.show{display:flex;align-items:center;gap:12px;}
.model-badge-icon{font-size:1.6rem;}
.model-badge-text p{font-size:.8rem;color:var(--muted);margin-top:2px;}
.model-badge-text strong{color:var(--green);font-size:.92rem;}

.upload-row{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px;}
.upload-box{position:relative;border:2px dashed var(--border);border-radius:12px;background:var(--surface2);
           cursor:pointer;aspect-ratio:1;display:flex;flex-direction:column;align-items:center;
           justify-content:center;gap:10px;transition:border-color .2s,background .2s;overflow:hidden;}
.upload-box:hover,.upload-box.drag-over{border-color:var(--accent);background:rgba(108,99,255,.05);}
.upload-box.has-image{border-color:var(--accent);border-style:solid;}
.upload-box input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
.upload-icon{font-size:2rem;}
.upload-label{font-size:.82rem;color:var(--muted);text-align:center;line-height:1.4;}
.upload-sub{font-size:.73rem;color:var(--border);}
.preview-img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:10px;}
.preview-overlay{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent,rgba(0,0,0,.6));
                 padding:12px 10px 10px;font-size:.73rem;color:#fff;text-align:center;opacity:0;transition:opacity .2s;}
.upload-box:hover .preview-overlay{opacity:1;}
.label-badge{position:absolute;top:10px;left:10px;background:rgba(108,99,255,.85);color:#fff;
            font-size:.7rem;font-weight:600;padding:3px 10px;border-radius:20px;letter-spacing:.04em;}

.btn{width:100%;padding:13px;font-size:.95rem;font-weight:600;border:none;border-radius:10px;
    cursor:pointer;transition:opacity .2s,transform .1s;}
.btn:active{transform:scale(.98);}
.btn:disabled{opacity:.35;cursor:not-allowed;}
.btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;}
.btn-primary:not(:disabled):hover{opacity:.9;}
.btn-reset{margin-top:8px;background:transparent;border:1px solid var(--border);color:var(--muted);
          font-size:.83rem;padding:8px;}
.btn-reset:hover{border-color:var(--muted);color:var(--text);}

.loading{display:none;text-align:center;padding:20px;color:var(--muted);}
.spinner{width:34px;height:34px;border:3px solid var(--border);border-top-color:var(--accent);
        border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 10px;}
@keyframes spin{to{transform:rotate(360deg);}}

#result{display:none;margin-top:24px;}
.result-faces{display:flex;gap:16px;margin-bottom:20px;align-items:center;justify-content:center;}
.face-thumb{text-align:center;}
.face-thumb img{width:88px;height:88px;object-fit:cover;border-radius:50%;border:3px solid var(--accent);}
.face-thumb p{font-size:.7rem;color:var(--muted);margin-top:5px;}
.vs-icon{font-size:1.3rem;color:var(--muted);}

.verdict-box{text-align:center;padding:22px 18px;border-radius:12px;margin-bottom:20px;border:1px solid transparent;}
.verdict-box.high{background:rgba(76,175,135,.1);border-color:rgba(76,175,135,.3);}
.verdict-box.medium{background:rgba(245,200,66,.08);border-color:rgba(245,200,66,.3);}
.verdict-box.low{background:rgba(245,138,66,.08);border-color:rgba(245,138,66,.3);}
.verdict-box.none{background:rgba(245,90,90,.08);border-color:rgba(245,90,90,.3);}
.verdict-emoji{font-size:2.2rem;margin-bottom:7px;}
.verdict-text{font-size:1.35rem;font-weight:700;letter-spacing:.04em;}
.verdict-sub{font-size:.85rem;color:var(--muted);margin-top:3px;}
.verdict-box.high .verdict-text{color:var(--green);}
.verdict-box.medium .verdict-text{color:var(--yellow);}
.verdict-box.low .verdict-text{color:var(--orange);}
.verdict-box.none .verdict-text{color:var(--red);}

.scores{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:18px;}
.score-item{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:14px 10px;text-align:center;}
.score-val{font-size:1.55rem;font-weight:700;color:var(--accent2);}
.score-label{font-size:.7rem;color:var(--muted);margin-top:4px;line-height:1.3;}
.progress-bar{height:5px;background:var(--border);border-radius:4px;margin-top:7px;overflow:hidden;}
.progress-fill{height:100%;border-radius:4px;transition:width 1s ease;}

.info-note{font-size:.76rem;color:var(--muted);text-align:center;padding:12px 14px;
          background:var(--surface2);border-radius:8px;border:1px solid var(--border);}
.warn-box{background:rgba(245,200,66,.08);border:1px solid rgba(245,200,66,.3);color:#e8c84a;
         border-radius:10px;padding:12px 16px;font-size:.85rem;margin-bottom:16px;display:none;}
.error-box{display:none;background:rgba(245,90,90,.1);border:1px solid rgba(245,90,90,.3);color:#ff8a8a;
          border-radius:10px;padding:13px 16px;font-size:.88rem;margin-top:14px;}
footer{margin-top:36px;font-size:.76rem;color:var(--muted);text-align:center;}
</style>
</head>
<body>

<header>
  <div class="logo"><div class="logo-icon">🧠</div><h1>Face Match ML</h1></div>
  <p class="subtitle">Training model PCA sendiri lalu bandingkan dua wajah</p>
</header>

<div class="tabs">
  <div class="tab active" id="tab-train" onclick="switchTab('train')">⚙️ 1. Training Model</div>
  <div class="tab" id="tab-compare" onclick="switchTab('compare')">🔍 2. Bandingkan Wajah</div>
</div>

<div class="card">

  <!-- SECTION TRAIN -->
  <div class="section active" id="sec-train">
    <div class="model-badge" id="modelBadge">
      <div class="model-badge-icon">✅</div>
      <div class="model-badge-text">
        <strong id="badgeTitle">Model sudah tertraining!</strong>
        <p id="badgeDetail">—</p>
      </div>
    </div>

    <div class="subtabs">
      <div class="subtab active" id="subtab-synthetic" onclick="switchSubtab('synthetic')">🎲 Dataset Sintetis</div>
      <div class="subtab" id="subtab-zip" onclick="switchSubtab('zip')">📦 Upload ZIP</div>
    </div>

    <!-- SUBSECTION: SYNTHETIC -->
    <div class="subsection active" id="sub-synthetic">
      <p style="font-size:.85rem;color:var(--muted);margin-bottom:18px;line-height:1.6;">
        Generate dataset sintetis otomatis. Setiap orang memiliki pola piksel unik + variasi noise,
        lalu PCA/SVD mereduksi dimensinya ke eigenfaces.
      </p>
      <div class="train-config">
        <div class="field">
          <label>Jumlah Orang: <span class="val" id="valPersons">5</span></label>
          <input type="range" id="slPersons" min="2" max="20" value="5"
                 oninput="document.getElementById('valPersons').textContent=this.value" />
        </div>
        <div class="field">
          <label>Foto per Orang: <span class="val" id="valImages">8</span></label>
          <input type="range" id="slImages" min="3" max="20" value="8"
                 oninput="document.getElementById('valImages').textContent=this.value" />
        </div>
      </div>
      <button class="btn btn-primary" id="trainSynthBtn" onclick="startTrainSynthetic()">🚀 Mulai Training (Sintetis)</button>
    </div>

    <!-- SUBSECTION: ZIP -->
    <div class="subsection" id="sub-zip">
      <div class="format-hint">
        📋 <strong>Format ZIP:</strong> semua foto rata (tanpa folder), nama file = <code>namaorang_nomor.ext</code><br>
        Contoh: <code>andi_1.jpg</code>, <code>andi_2.jpg</code>, <code>budi_1.png</code>, <code>siti_wulan_1.jpg</code><br>
        Minimal 2 orang berbeda, format gambar: jpg / jpeg / png / bmp / webp
      </div>

      <div class="dropzone" id="zipDropzone" onclick="document.getElementById('zipInput').click()">
        <input type="file" id="zipInput" accept=".zip" onchange="handleZipFile(event)" />
        <div class="dropzone-icon">📦</div>
        <div class="dropzone-label">Klik atau drag & drop file ZIP dataset di sini</div>
        <div class="dropzone-sub">Maksimal ukuran sesuai konfigurasi server</div>
        <div class="file-chosen" id="zipFileChosen"></div>
      </div>

      <button class="btn btn-primary" id="trainZipBtn" onclick="startTrainZip()" disabled>🚀 Mulai Training (dari ZIP)</button>
    </div>

    <div class="log-box" id="logBox"></div>
  </div>

  <!-- SECTION COMPARE -->
  <div class="section" id="sec-compare">
    <div class="warn-box" id="warnNoModel">
      ⚠️ Model belum ditraining. Pergi ke tab <strong>Training Model</strong> terlebih dahulu.
    </div>
    <div class="upload-row">
      <div class="upload-box" id="box1" onclick="triggerUpload(1)">
        <input type="file" id="file1" accept="image/*" onchange="handleFile(event,1)" onclick="event.stopPropagation()" />
        <div class="upload-icon">📷</div>
        <div class="upload-label">Foto Pertama<br><span class="upload-sub">JPG · PNG · WEBP</span></div>
        <img class="preview-img" id="preview1" style="display:none" />
        <div class="preview-overlay">Klik untuk ganti</div>
        <span class="label-badge" id="badge1" style="display:none">Foto 1</span>
      </div>
      <div class="upload-box" id="box2" onclick="triggerUpload(2)">
        <input type="file" id="file2" accept="image/*" onchange="handleFile(event,2)" onclick="event.stopPropagation()" />
        <div class="upload-icon">📸</div>
        <div class="upload-label">Foto Kedua<br><span class="upload-sub">JPG · PNG · WEBP</span></div>
        <img class="preview-img" id="preview2" style="display:none" />
        <div class="preview-overlay">Klik untuk ganti</div>
        <span class="label-badge" id="badge2" style="display:none">Foto 2</span>
      </div>
    </div>
    <button class="btn btn-primary" id="analyzeBtn" onclick="analyze()" disabled>🔎 Analisis Kemiripan</button>
    <button class="btn btn-reset" onclick="resetCompare()">↺ Reset Foto</button>
    <div class="loading" id="loading">
      <div class="spinner"></div>
      <p>Memproyeksikan wajah ke ruang PCA...</p>
    </div>
    <div class="error-box" id="errorBox"></div>
    <div id="result">
      <hr style="border-color:var(--border);margin:22px 0 20px;" />
      <div class="result-faces">
        <div class="face-thumb"><img id="thumb1" src="" /><p>Wajah 1</p></div>
        <div class="vs-icon">⟺</div>
        <div class="face-thumb"><img id="thumb2" src="" /><p>Wajah 2</p></div>
      </div>
      <div class="verdict-box" id="verdictBox">
        <div class="verdict-emoji" id="verdictEmoji"></div>
        <div class="verdict-text" id="verdictText"></div>
        <div class="verdict-sub" id="verdictSub"></div>
      </div>
      <div class="scores">
        <div class="score-item">
          <div class="score-val" id="scoreCombined">—</div>
          <div class="progress-bar"><div class="progress-fill" id="barCombined" style="width:0;background:var(--accent)"></div></div>
          <div class="score-label">Skor Gabungan</div>
        </div>
        <div class="score-item">
          <div class="score-val" id="scoreCosine">—</div>
          <div class="progress-bar"><div class="progress-fill" id="barCosine" style="width:0;background:var(--accent2)"></div></div>
          <div class="score-label">Cosine Similarity</div>
        </div>
        <div class="score-item">
          <div class="score-val" id="scoreEuclid">—</div>
          <div class="progress-bar"><div class="progress-fill" id="barEuclid" style="width:0;background:#9f97ff"></div></div>
          <div class="score-label">Euclidean Sim</div>
        </div>
      </div>
      <div class="info-note" id="infoNote">
        Analisis menggunakan PCA/SVD Eigenfaces. Skor 85%+ sangat mirip | 70-85% mungkin sama | di bawah 70% berbeda.
      </div>
    </div>
  </div>

</div>
<footer>Berbasis face_recognation.py · PCA/SVD Eigenfaces · Flask + OpenCV · Single-file app</footer>

<script>
(function () {
  var images = { 1: null, 2: null };
  var modelTrained = false;
  var pollTimer = null;
  var zipFile = null;

  fetch('/model_info').then(function(r){ return r.json(); }).then(function(d){
    if (d.trained) { modelTrained = true; showBadge(d); }
  }).catch(function(){});

  window.switchTab = function(tab) {
    document.querySelectorAll('.tab, .section').forEach(function(el){ el.classList.remove('active'); });
    document.getElementById('tab-' + tab).classList.add('active');
    document.getElementById('sec-' + tab).classList.add('active');
    if (tab === 'compare') {
      document.getElementById('warnNoModel').style.display = modelTrained ? 'none' : 'block';
    }
  };

  window.switchSubtab = function(sub) {
    document.querySelectorAll('.subtab, .subsection').forEach(function(el){ el.classList.remove('active'); });
    document.getElementById('subtab-' + sub).classList.add('active');
    document.getElementById('sub-' + sub).classList.add('active');
  };

  function pollTrainStatus(onDone) {
    var logBox = document.getElementById('logBox');
    logBox.style.display = 'block';
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(function () {
      fetch('/train_status').then(function(r){ return r.json(); }).then(function(d){
        logBox.textContent = d.log;
        logBox.scrollTop = logBox.scrollHeight;
        if (d.trained) {
          clearInterval(pollTimer);
          modelTrained = true;
          fetch('/model_info').then(function(r){ return r.json(); }).then(showBadge);
          onDone(true);
        } else if (d.log.indexOf('Error:') !== -1) {
          clearInterval(pollTimer);
          onDone(false);
        }
      });
    }, 800);
  }

  window.startTrainSynthetic = function() {
    var nPersons = parseInt(document.getElementById('slPersons').value);
    var nImages = parseInt(document.getElementById('slImages').value);
    var btn = document.getElementById('trainSynthBtn');

    btn.disabled = true; btn.textContent = '⏳ Training...';
    document.getElementById('logBox').textContent = 'Memulai training...\n';
    document.getElementById('modelBadge').classList.remove('show');
    modelTrained = false;

    fetch('/train_synthetic', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ n_persons: nPersons, n_images: nImages })
    }).then(function(){
      pollTrainStatus(function(){
        btn.disabled = false; btn.textContent = '🚀 Mulai Training (Sintetis)';
      });
    });
  };

  window.handleZipFile = function(event) {
    var file = event.target.files[0];
    if (!file) return;
    zipFile = file;
    document.getElementById('zipDropzone').classList.add('has-file');
    document.getElementById('zipFileChosen').textContent = '✓ ' + file.name + ' (' + (file.size/1024/1024).toFixed(2) + ' MB)';
    document.getElementById('trainZipBtn').disabled = false;
  };

  window.startTrainZip = function() {
    if (!zipFile) return;
    var btn = document.getElementById('trainZipBtn');
    btn.disabled = true; btn.textContent = '⏳ Training...';
    document.getElementById('logBox').textContent = 'Mengupload dan memproses ZIP...\n';
    document.getElementById('modelBadge').classList.remove('show');
    modelTrained = false;

    var formData = new FormData();
    formData.append('dataset', zipFile);

    fetch('/train_zip', { method: 'POST', body: formData })
      .then(function(r){ return r.json(); })
      .then(function(resp){
        if (resp.error) {
          document.getElementById('logBox').textContent = 'Error: ' + resp.error;
          btn.disabled = false; btn.textContent = '🚀 Mulai Training (dari ZIP)';
          return;
        }
        pollTrainStatus(function(){
          btn.disabled = false; btn.textContent = '🚀 Mulai Training (dari ZIP)';
        });
      });
  };

  var zipDz = document.getElementById('zipDropzone');
  zipDz.addEventListener('dragover', function(e){ e.preventDefault(); zipDz.classList.add('drag-over'); });
  zipDz.addEventListener('dragleave', function(){ zipDz.classList.remove('drag-over'); });
  zipDz.addEventListener('drop', function(e){
    e.preventDefault(); zipDz.classList.remove('drag-over');
    var file = e.dataTransfer.files[0];
    if (!file || !file.name.toLowerCase().endsWith('.zip')) return;
    zipFile = file;
    zipDz.classList.add('has-file');
    document.getElementById('zipFileChosen').textContent = '✓ ' + file.name + ' (' + (file.size/1024/1024).toFixed(2) + ' MB)';
    document.getElementById('trainZipBtn').disabled = false;
  });

  function showBadge(d) {
    document.getElementById('badgeDetail').textContent =
      d.n_persons + ' orang · ' + d.n_samples + ' sampel · ' + d.n_components + ' komponen PCA · ' + d.explained_variance + '% explained variance';
    document.getElementById('modelBadge').classList.add('show');
  }

  window.triggerUpload = function(n) { document.getElementById('file' + n).click(); };

  window.handleFile = function(event, n) {
    event.stopPropagation();
    var file = event.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function(e) {
      images[n] = e.target.result;
      var p = document.getElementById('preview' + n);
      p.src = e.target.result; p.style.display = 'block';
      document.getElementById('badge' + n).style.display = 'inline-block';
      document.getElementById('box' + n).classList.add('has-image');
      checkReady();
    };
    reader.readAsDataURL(file);
  };

  function checkReady() {
    document.getElementById('analyzeBtn').disabled = !(images[1] && images[2] && modelTrained);
  }

  window.analyze = function() {
    hideError();
    document.getElementById('result').style.display = 'none';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('analyzeBtn').disabled = true;

    fetch('/compare', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ image1: images[1], image2: images[2] })
    }).then(function(r){ return r.json(); }).then(function(data){
      document.getElementById('loading').style.display = 'none';
      if (data.error) { showError(data.error); } else { renderResult(data); }
      document.getElementById('analyzeBtn').disabled = false;
    }).catch(function(err){
      document.getElementById('loading').style.display = 'none';
      showError('Koneksi gagal: ' + err.message);
      document.getElementById('analyzeBtn').disabled = false;
    });
  };

  function renderResult(d) {
    document.getElementById('thumb1').src = d.thumbnail1;
    document.getElementById('thumb2').src = d.thumbnail2;
    var emojis = { high:'✅', medium:'🤔', low:'⚠️', none:'❌' };
    var box = document.getElementById('verdictBox');
    box.className = 'verdict-box ' + d.verdict_level;
    document.getElementById('verdictEmoji').textContent = emojis[d.verdict_level];
    document.getElementById('verdictText').textContent = d.verdict;
    document.getElementById('verdictSub').textContent = d.verdict_sub;
    document.getElementById('scoreCombined').textContent = d.combined_score + '%';
    document.getElementById('scoreCosine').textContent = d.cosine_similarity + '%';
    document.getElementById('scoreEuclid').textContent = d.euclidean_sim + '%';
    document.getElementById('infoNote').textContent =
      'Model: ' + d.n_persons + ' orang, ' + d.n_train_samples + ' sampel training — PCA/SVD Eigenfaces';
    setTimeout(function() {
      document.getElementById('barCombined').style.width = d.combined_score + '%';
      document.getElementById('barCosine').style.width = d.cosine_similarity + '%';
      document.getElementById('barEuclid').style.width = d.euclidean_sim + '%';
    }, 100);
    document.getElementById('result').style.display = 'block';
    document.getElementById('result').scrollIntoView({ behavior:'smooth', block:'nearest' });
  }

  function showError(msg) {
    var el = document.getElementById('errorBox');
    el.textContent = '⚠️ ' + msg; el.style.display = 'block';
  }
  function hideError() { document.getElementById('errorBox').style.display = 'none'; }

  window.resetCompare = function() {
    images = { 1: null, 2: null };
    [1,2].forEach(function(n) {
      document.getElementById('file'+n).value = '';
      document.getElementById('preview'+n).style.display = 'none';
      document.getElementById('badge'+n).style.display = 'none';
      document.getElementById('box'+n).classList.remove('has-image');
    });
    document.getElementById('result').style.display = 'none';
    document.getElementById('loading').style.display = 'none';
    document.getElementById('analyzeBtn').disabled = true;
    hideError();
  };

  [1,2].forEach(function(n) {
    var box = document.getElementById('box'+n);
    box.addEventListener('dragover', function(e){ e.preventDefault(); box.classList.add('drag-over'); });
    box.addEventListener('dragleave', function(){ box.classList.remove('drag-over'); });
    box.addEventListener('drop', function(e){
      e.preventDefault(); box.classList.remove('drag-over');
      var file = e.dataTransfer.files[0];
      if (!file || !file.type.startsWith('image/')) return;
      var reader = new FileReader();
      reader.onload = function(ev) {
        images[n] = ev.target.result;
        var p = document.getElementById('preview'+n);
        p.src = ev.target.result; p.style.display = 'block';
        document.getElementById('badge'+n).style.display = 'inline-block';
        box.classList.add('has-image');
        checkReady();
      };
      reader.readAsDataURL(file);
    });
  });

})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=5000)
