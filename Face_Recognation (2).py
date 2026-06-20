"""
Face Match ML — Streamlit App
==============================
Aplikasi web untuk training model PCA/SVD dari dataset wajah (sintetis atau upload ZIP),
lalu membandingkan dua foto untuk menentukan apakah orang yang sama.

Cara jalankan lokal:
    pip install streamlit opencv-python-headless numpy scikit-learn pillow
    streamlit run app.py

Deploy ke Streamlit Community Cloud:
    1. Push app.py + requirements.txt ke GitHub repo
    2. Buka share.streamlit.io -> New app -> pilih repo & app.py
    3. Selesai, Streamlit Cloud otomatis install requirements.txt & jalankan

Format ZIP dataset:
    Semua foto rata (tanpa folder), nama file = "namaorang_nomor.ext"
    Contoh: andi_1.jpg, andi_2.jpg, budi_1.jpg, budi_2.png, siti_1.jpeg ...
    Nama orang diambil dari bagian sebelum underscore + angka terakhir.
"""

import streamlit as st
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from PIL import Image
import io, os, re, zipfile, tempfile, shutil

# ── Konfigurasi ──────────────────────────────
IMG_SIZE     = (100, 100)
N_COMPONENTS = 50
ALLOWED_EXT  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

st.set_page_config(
    page_title="Face Match ML",
    page_icon="🧠",
    layout="centered",
)


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def preprocess_image(img: np.ndarray, use_face_detection=True) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_face_detection:
        cascade = get_face_cascade()
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            gray = gray[y:y + h, x:x + w]
    return cv2.resize(gray, IMG_SIZE).astype(np.float64).flatten() / 255.0


def get_face_crop(img: np.ndarray) -> np.ndarray:
    """Crop wajah untuk preview (return BGR array, bukan base64)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    crop = img
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        pad = int(min(w, h) * 0.15)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (200, 200))


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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

def create_synthetic_dataset(n_persons=5, n_images=8, log_fn=None):
    """Buat dataset sintetis: setiap orang punya 'wajah dasar' unik + variasi noise."""
    if log_fn: log_fn("Membuat dataset sintetis...")
    X, labels = [], []
    names = [f"orang_{i+1}" for i in range(n_persons)]

    for name in names:
        rng = np.random.default_rng(abs(hash(name)) % 2**31)
        base_face = rng.integers(40, 220, IMG_SIZE).astype(np.float64)

        cx, cy = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2
        for px in range(IMG_SIZE[0]):
            for py in range(IMG_SIZE[1]):
                dist = ((px - cx) / 40) ** 2 + ((py - cy) / 50) ** 2
                if dist < 1.0:
                    base_face[px, py] *= 0.75

        for j in range(n_images):
            noise = rng.integers(-25, 26, IMG_SIZE).astype(np.float64)
            face = np.clip(base_face + noise, 0, 255) / 255.0
            X.append(face.flatten())
            labels.append(name)

        if log_fn: log_fn(f"  ✓ {name}: {n_images} gambar dibuat")

    return np.array(X), np.array(labels)


# ─────────────────────────────────────────────
# DATASET DARI ZIP UPLOAD
# ─────────────────────────────────────────────

def load_dataset_from_zip(zip_bytes: bytes, log_fn=None):
    """
    Ekstrak ZIP (dari bytes di memori), baca semua gambar (flat scan ke semua subfolder),
    label diambil dari nama file pakai extract_label_from_filename().
    """
    if log_fn: log_fn("Mengekstrak dan membaca dataset dari ZIP...")

    extract_dir = tempfile.mkdtemp(prefix="faceds_")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
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
        if log_fn:
            log_fn(f"  ✓ {len(X)} gambar berhasil dibaca, {len(unique_labels)} orang terdeteksi")
            if skipped > 0:
                log_fn(f"  ⚠ {skipped} file dilewati (gagal dibaca / format tidak didukung)")
            for lbl in unique_labels:
                count = labels.count(lbl)
                log_fn(f"    - {lbl}: {count} foto")

        return np.array(X), np.array(labels)

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# TRAINING PCA
# ─────────────────────────────────────────────

def run_training(X: np.ndarray, labels: np.ndarray, log_fn=None):
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        raise ValueError("Minimal perlu 2 orang berbeda dalam dataset untuk training")
    if X.shape[0] < 3:
        raise ValueError("Minimal perlu 3 foto total dalam dataset untuk training")

    if log_fn: log_fn(f"\nDataset: {X.shape[0]} gambar | {n_unique} orang | dimensi {X.shape[1]}")

    n_comp = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1])
    if log_fn: log_fn(f"\nTraining PCA dengan {n_comp} komponen...")

    pca = PCA(n_components=n_comp, svd_solver="full")
    X_pca = pca.fit_transform(X)

    explained = float(np.sum(pca.explained_variance_ratio_)) * 100
    if log_fn:
        log_fn(f"  ✓ Explained variance: {explained:.1f}%")
        log_fn(f"  ✓ Dimensi PCA: {X_pca.shape}")
        log_fn(f"\n✅ Model berhasil ditraining! Siap membandingkan wajah.")

    return {"pca": pca, "X_pca": X_pca, "labels": labels}


def compare_faces(model: dict, img1: np.ndarray, img2: np.ndarray) -> dict:
    vec1 = preprocess_image(img1)
    vec2 = preprocess_image(img2)

    pca = model["pca"]
    z1 = pca.transform(vec1.reshape(1, -1))
    z2 = pca.transform(vec2.reshape(1, -1))

    cos_sim = float(cosine_similarity(z1, z2)[0][0])
    euc_dist = float(euclidean_distances(z1, z2)[0][0])
    euc_sim = max(0, 1.0 - min(euc_dist / 30.0, 1.0))
    combined = cos_sim * 0.6 + euc_sim * 0.4

    if combined >= 0.85:
        verdict, sub, level, emoji = "SANGAT MIRIP", "Kemungkinan besar orang yang sama", "high", "✅"
    elif combined >= 0.70:
        verdict, sub, level, emoji = "CUKUP MIRIP", "Mungkin orang yang sama", "medium", "🤔"
    elif combined >= 0.50:
        verdict, sub, level, emoji = "KURANG MIRIP", "Kemungkinan orang berbeda", "low", "⚠️"
    else:
        verdict, sub, level, emoji = "TIDAK MIRIP", "Kemungkinan besar orang berbeda", "none", "❌"

    return {
        "cosine_similarity": round(cos_sim * 100, 1),
        "euclidean_sim": round(euc_sim * 100, 1),
        "combined_score": round(combined * 100, 1),
        "verdict": verdict,
        "verdict_sub": sub,
        "verdict_level": level,
        "emoji": emoji,
    }


# ─────────────────────────────────────────────
# UI — STYLING
# ─────────────────────────────────────────────

st.markdown("""
<style>
.verdict-box {
    text-align: center; padding: 24px 18px; border-radius: 14px;
    margin: 16px 0; border: 1px solid transparent;
}
.verdict-box.high   { background: rgba(76,175,135,.12);  border-color: rgba(76,175,135,.4); }
.verdict-box.medium { background: rgba(245,200,66,.10);  border-color: rgba(245,200,66,.4); }
.verdict-box.low    { background: rgba(245,138,66,.10);  border-color: rgba(245,138,66,.4); }
.verdict-box.none   { background: rgba(245,90,90,.10);   border-color: rgba(245,90,90,.4); }
.verdict-emoji { font-size: 2.4rem; }
.verdict-text  { font-size: 1.5rem; font-weight: 700; letter-spacing: .03em; margin-top: 4px; }
.verdict-sub   { font-size: .9rem; opacity: .75; margin-top: 4px; }
.verdict-box.high   .verdict-text { color: #4caf87; }
.verdict-box.medium .verdict-text { color: #d9a627; }
.verdict-box.low    .verdict-text { color: #e07b2e; }
.verdict-box.none   .verdict-text { color: #e0504f; }
.format-hint {
    font-size: .85rem; background: rgba(108,99,255,.08); border: 1px solid rgba(108,99,255,.25);
    border-radius: 10px; padding: 12px 16px; line-height: 1.6;
}
.format-hint code { background: rgba(108,99,255,.18); padding: 1px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UI — SESSION STATE
# ─────────────────────────────────────────────

if "model" not in st.session_state:
    st.session_state.model = None
if "train_log" not in st.session_state:
    st.session_state.train_log = []


def log(msg):
    st.session_state.train_log.append(msg)


# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────

st.markdown("## 🧠 Face Match ML")
st.caption("Training model PCA sendiri, lalu bandingkan dua wajah — apakah orang yang sama atau bukan.")

tab_train, tab_compare = st.tabs(["⚙️ 1. Training Model", "🔍 2. Bandingkan Wajah"])


# ─────────────────────────────────────────────
# TAB 1 — TRAINING
# ─────────────────────────────────────────────

with tab_train:

    if st.session_state.model is not None:
        m = st.session_state.model
        n_persons = len(np.unique(m["labels"]))
        n_samples = len(m["labels"])
        explained = float(np.sum(m["pca"].explained_variance_ratio_)) * 100
        st.success(
            f"✅ **Model sudah tertraining!**  \n"
            f"{n_persons} orang · {n_samples} sampel · {m['pca'].n_components_} komponen PCA · "
            f"{explained:.1f}% explained variance"
        )

    method = st.radio(
        "Pilih metode dataset",
        ["🎲 Dataset Sintetis", "📦 Upload ZIP"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.write("")

    if method == "🎲 Dataset Sintetis":
        st.caption(
            "Generate dataset sintetis otomatis. Setiap orang punya pola piksel unik + variasi noise, "
            "lalu PCA/SVD mereduksi dimensinya ke eigenfaces. Cocok untuk uji coba cepat tanpa dataset asli."
        )
        col1, col2 = st.columns(2)
        with col1:
            n_persons = st.slider("Jumlah Orang", 2, 20, 5)
        with col2:
            n_images = st.slider("Foto per Orang", 3, 20, 8)

        if st.button("🚀 Mulai Training (Sintetis)", type="primary", use_container_width=True):
            st.session_state.train_log = []
            log_box = st.empty()
            try:
                with st.spinner("Training berjalan..."):
                    X, labels = create_synthetic_dataset(n_persons, n_images, log_fn=log)
                    log_box.code("\n".join(st.session_state.train_log))
                    model = run_training(X, labels, log_fn=log)
                    log_box.code("\n".join(st.session_state.train_log))
                st.session_state.model = model
                st.rerun()
            except Exception as e:
                log(f"\n❌ Error: {e}")
                log_box.code("\n".join(st.session_state.train_log))
                st.error(str(e))

    else:  # Upload ZIP
        st.markdown("""
        <div class="format-hint">
        📋 <strong>Format ZIP:</strong> semua foto rata (tanpa folder), nama file = <code>namaorang_nomor.ext</code><br>
        Contoh: <code>andi_1.jpg</code>, <code>andi_2.jpg</code>, <code>budi_1.png</code>, <code>siti_wulan_1.jpg</code><br>
        Minimal 2 orang berbeda. Format gambar yang didukung: jpg, jpeg, png, bmp, webp
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        zip_file = st.file_uploader("Upload file ZIP dataset", type=["zip"])

        if st.button("🚀 Mulai Training (dari ZIP)", type="primary",
                      use_container_width=True, disabled=(zip_file is None)):
            st.session_state.train_log = []
            log_box = st.empty()
            try:
                with st.spinner("Mengekstrak dan training..."):
                    zip_bytes = zip_file.read()
                    X, labels = load_dataset_from_zip(zip_bytes, log_fn=log)
                    log_box.code("\n".join(st.session_state.train_log))
                    model = run_training(X, labels, log_fn=log)
                    log_box.code("\n".join(st.session_state.train_log))
                st.session_state.model = model
                st.rerun()
            except Exception as e:
                log(f"\n❌ Error: {e}")
                log_box.code("\n".join(st.session_state.train_log))
                st.error(str(e))

    if st.session_state.train_log and st.session_state.model is None:
        with st.expander("📜 Log training terakhir", expanded=True):
            st.code("\n".join(st.session_state.train_log))


# ─────────────────────────────────────────────
# TAB 2 — COMPARE
# ─────────────────────────────────────────────

with tab_compare:

    if st.session_state.model is None:
        st.warning("⚠️ Model belum ditraining. Buka tab **Training Model** terlebih dahulu.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            file1 = st.file_uploader("Foto Pertama", type=["jpg", "jpeg", "png", "webp"], key="up1")
            if file1:
                st.image(file1, use_container_width=True)
        with col2:
            file2 = st.file_uploader("Foto Kedua", type=["jpg", "jpeg", "png", "webp"], key="up2")
            if file2:
                st.image(file2, use_container_width=True)

        analyze_disabled = not (file1 and file2)
        if st.button("🔎 Analisis Kemiripan", type="primary",
                      use_container_width=True, disabled=analyze_disabled):
            try:
                with st.spinner("Memproyeksikan wajah ke ruang PCA..."):
                    pil1 = Image.open(file1)
                    pil2 = Image.open(file2)
                    img1 = pil_to_bgr(pil1)
                    img2 = pil_to_bgr(pil2)

                    crop1 = get_face_crop(img1)
                    crop2 = get_face_crop(img2)

                    result = compare_faces(st.session_state.model, img1, img2)

                st.markdown("---")

                fc1, fc2, fc3 = st.columns([1, 0.3, 1])
                with fc1:
                    st.image(bgr_to_rgb(crop1), caption="Wajah 1", use_container_width=True)
                with fc2:
                    st.markdown(
                        "<div style='text-align:center;font-size:1.8rem;padding-top:60px;'>⟺</div>",
                        unsafe_allow_html=True,
                    )
                with fc3:
                    st.image(bgr_to_rgb(crop2), caption="Wajah 2", use_container_width=True)

                st.markdown(f"""
                <div class="verdict-box {result['verdict_level']}">
                    <div class="verdict-emoji">{result['emoji']}</div>
                    <div class="verdict-text">{result['verdict']}</div>
                    <div class="verdict-sub">{result['verdict_sub']}</div>
                </div>
                """, unsafe_allow_html=True)

                s1, s2, s3 = st.columns(3)
                s1.metric("Skor Gabungan", f"{result['combined_score']}%")
                s2.metric("Cosine Similarity", f"{result['cosine_similarity']}%")
                s3.metric("Euclidean Sim", f"{result['euclidean_sim']}%")

                n_persons = len(np.unique(st.session_state.model["labels"]))
                n_samples = len(st.session_state.model["labels"])
                st.caption(
                    f"⚙️ Model: {n_persons} orang, {n_samples} sampel training — PCA/SVD Eigenfaces.  "
                    f"Skor ≥85% sangat mirip · 70-85% mungkin sama · <70% kemungkinan berbeda."
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

        if st.button("↺ Reset Foto"):
            st.session_state.pop("up1", None)
            st.session_state.pop("up2", None)
            st.rerun()


st.markdown("---")
st.caption("Berbasis face_recognation.py · PCA/SVD Eigenfaces · Streamlit + OpenCV")
