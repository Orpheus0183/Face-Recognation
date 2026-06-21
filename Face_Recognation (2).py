"""
Face Match ML — Streamlit App
==============================
Aplikasi web untuk training model PCA/SVD dari dataset wajah (sintetis atau upload ZIP),
lalu membandingkan dua foto untuk menentukan apakah orang yang sama.

Cara jalankan lokal:
    pip install streamlit opencv-python-headless numpy scikit-learn pillow py7zr
    streamlit run app.py

Deploy ke Streamlit Community Cloud:
    1. Push app.py + requirements.txt (+ .streamlit/config.toml) ke GitHub repo
    2. Buka share.streamlit.io -> New app -> pilih repo & app.py
    3. Selesai, Streamlit Cloud otomatis install requirements.txt & jalankan

Format arsip dataset yang didukung: .zip dan .7z (dua mode struktur, dideteksi otomatis):
    1. Folder per orang:
       dataset/andi/foto1.jpg, dataset/andi/foto2.jpg
       dataset/budi/apapun_namanya.png
       -> Nama folder = label orang, nama file di dalamnya bebas.

    2. File rata (flat):
       dataset/andi_1.jpg, dataset/budi_1.jpg
       -> Label diambil dari nama file sebelum angka terakhir.
"""

import streamlit as st
import cv2
import numpy as np
import requests
from skimage.metrics import structural_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from PIL import Image
import io, os, re, time, zipfile, tempfile, shutil

try:
    import py7zr
    PY7ZR_AVAILABLE = True
except ImportError:
    PY7ZR_AVAILABLE = False

# ── Konfigurasi ──────────────────────────────
IMG_SIZE       = (80, 80)     # disesuaikan dgn resolusi sumber foto publik figur (~50-60px asli)
N_COMPONENTS   = 60           # diturunkan sedikit mengikuti dimensi fitur yg juga lebih kecil
ALLOWED_EXT    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Parameter LBP (Local Binary Pattern) -> fitur tekstur tahan-cahaya
LBP_RADIUS     = 2
LBP_N_POINTS   = 8 * LBP_RADIUS
LBP_GRID       = (6, 6)       # grid lebih kecil krn IMG_SIZE juga lebih kecil (80x80)

# Parameter Modular PCA (Gottumukkal & Asari, 2004) -> PCA terpisah per
# sub-region wajah, supaya variasi lokal (ekspresi) tidak mendistorsi
# representasi area lain yang stabil (mata, hidung).
N_COMPONENTS_PER_MODULE = 12   # komponen PCA per modul (modul lebih kecil dari wajah utuh)
N_COMPONENTS_LBP        = 30   # komponen PCA untuk fitur LBP global
N_COMPONENTS_HOG        = 30   # komponen PCA untuk fitur HOG global

# Eigenface discarding (Turk & Pentland, 1991; Wikipedia "Eigenface") ->
# beberapa komponen PCA pertama biasanya didominasi variasi pencahayaan
# global, bukan informasi identitas wajah, sehingga literatur klasik
# menyarankan membuangnya. Diuji coba empiris pada dataset ini (lihat log
# eksperimen): discard=0 memberi skor similarity tertinggi dibanding
# discard=1/2/3, kemungkinan karena foto sumber sudah melalui CLAHE
# (equalization) sehingga variasi pencahayaan sudah banyak terkompensasi
# di awal -- membuang komponen tambahan justru membuang informasi identitas.
N_EIGENFACE_DISCARD = 0

st.set_page_config(
    page_title="Face Match ML",
    page_icon="🧠",
    layout="centered",
)


# ─────────────────────────────────────────────
# PREPROCESSING — FACE ALIGNMENT (Haar Cascade)
# ─────────────────────────────────────────────
#
# Catatan: versi sebelumnya sempat mencoba mediapipe Face Mesh (468 landmark)
# untuk alignment presisi tinggi. Secara empiris terbukti TIDAK meningkatkan
# skor similarity dibanding Haar Cascade pada dataset ini (mediapipe: 32.2%
# vs Haar Cascade: 54.5% untuk pasangan foto uji yang sama), dan mediapipe
# belum punya wheel terkompilasi untuk Python 3.14 (versi yang dipakai
# Streamlit Community Cloud), sehingga deployment gagal. Maka alignment
# dikembalikan ke Haar Cascade (Viola-Jones, 2001) yang juga merupakan
# metode deteksi wajah klasik yang valid secara akademik.

@st.cache_resource(show_spinner=False)
def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@st.cache_resource(show_spinner=False)
def get_eye_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def detect_face_box(gray: np.ndarray):
    """Cari kotak wajah terbesar. Return (x,y,w,h) atau None."""
    cascade = get_face_cascade()
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]


def align_face(gray: np.ndarray, face_box) -> np.ndarray:
    """
    Luruskan wajah berdasarkan posisi dua mata, supaya foto dengan kepala
    miring/beda sudut bisa dibandingkan secara piksel-sejajar dengan foto frontal.
    Kalau mata tidak terdeteksi, kembalikan crop wajah apa adanya (fallback).
    """
    x, y, w, h = face_box
    face_roi = gray[y:y + h, x:x + w]

    eye_cascade = get_eye_cascade()
    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5, minSize=(int(w*0.12), int(h*0.12)))

    if len(eyes) < 2:
        return face_roi  # fallback: tanpa alignment

    # Ambil 2 mata terbesar, urutkan kiri-kanan
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    eyes = sorted(eyes, key=lambda e: e[0])
    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes

    # Titik tengah tiap mata (koordinat relatif terhadap face_roi)
    left_eye  = (ex1 + ew1 // 2, ey1 + eh1 // 2)
    right_eye = (ex2 + ew2 // 2, ey2 + eh2 // 2)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    if dx == 0:
        return face_roi
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotasi seluruh gambar grayscale di sekitar pusat wajah, lalu crop ulang
    center = (float(x + w // 2), float(y + h // 2))
    rot_mat = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    rotated = cv2.warpAffine(gray, rot_mat, (gray.shape[1], gray.shape[0]),
                              flags=cv2.INTER_LINEAR)

    # Deteksi ulang wajah pada gambar yang sudah diluruskan (lebih akurat)
    aligned_box = detect_face_box(rotated)
    if aligned_box is not None:
        ax, ay, aw, ah = aligned_box
        return rotated[ay:ay + ah, ax:ax + aw]

    return rotated[y:y + h, x:x + w]


def normalize_lighting(gray: np.ndarray) -> np.ndarray:
    """Histogram equalization (CLAHE) -> menyamakan kontras/pencahayaan antar foto."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def get_aligned_face(img: np.ndarray, use_face_detection=True) -> np.ndarray:
    """
    Pipeline lengkap: grayscale -> deteksi wajah (Haar Cascade) -> alignment
    (mata) -> equalization -> resize ke IMG_SIZE. Return grayscale 2D array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if use_face_detection:
        face_box = detect_face_box(gray)
        if face_box is not None:
            gray = align_face(gray, face_box)

    gray = cv2.resize(gray, IMG_SIZE)
    gray = normalize_lighting(gray)
    return gray


# ─────────────────────────────────────────────
# PREPROCESSING — FITUR: PIKSEL + LBP
# ─────────────────────────────────────────────

def compute_lbp(gray: np.ndarray, radius=LBP_RADIUS, n_points=LBP_N_POINTS) -> np.ndarray:
    """
    Local Binary Pattern manual (tanpa dependency skimage).
    Untuk tiap piksel, bandingkan dengan n_points tetangga di sekeliling radius
    -> encode sebagai pola biner -> hasil tahan terhadap perubahan pencahayaan
    karena yang dibandingkan adalah relasi terang/gelap, bukan nilai absolut.
    """
    h, w = gray.shape
    # n_points bisa sampai 16 (radius=2) -> nilai LBP maksimum 2^16-1,
    # jadi pakai uint32 supaya tidak overflow.
    lbp = np.zeros((h, w), dtype=np.uint32)
    gray_f = gray.astype(np.float64)

    angles = [2 * np.pi * p / n_points for p in range(n_points)]
    offsets = [(radius * np.cos(a), -radius * np.sin(a)) for a in angles]

    padded = np.pad(gray_f, radius, mode="edge")

    for p, (dx, dy) in enumerate(offsets):
        map_x = (np.arange(w) + radius + dx).astype(np.float32)
        map_y = (np.arange(h) + radius + dy).astype(np.float32)
        map_x, map_y = np.meshgrid(map_x, map_y)
        sampled = cv2.remap(padded.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
        bit = (sampled >= gray_f).astype(np.uint32)
        lbp += (bit << p)

    return lbp


def lbp_histogram_features(gray: np.ndarray, grid=LBP_GRID) -> np.ndarray:
    """
    Hitung histogram LBP per sel grid, lalu gabungkan jadi satu vektor fitur.
    Nilai LBP mentah (0 .. 2^n_points-1) di-bin ulang ke 256 bucket supaya
    panjang vektor fitur tetap terkendali walau n_points besar.
    """
    lbp = compute_lbp(gray)
    h, w = lbp.shape
    gh, gw = grid
    cell_h, cell_w = h // gh, w // gw

    max_val = 2 ** LBP_N_POINTS
    n_bins = min(max_val, 256)
    # Skala nilai LBP mentah ke rentang [0, n_bins) sebelum histogram
    lbp_scaled = (lbp.astype(np.float64) / max_val * n_bins).astype(np.int64)
    lbp_scaled = np.clip(lbp_scaled, 0, n_bins - 1)

    hist_all = []
    for i in range(gh):
        for j in range(gw):
            cell = lbp_scaled[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            hist, _ = np.histogram(cell.flatten(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(np.float64)
            hist /= (hist.sum() + 1e-7)  # normalisasi per sel
            hist_all.append(hist)

    return np.concatenate(hist_all)


def preprocess_image(img: np.ndarray, use_face_detection=True) -> np.ndarray:
    """
    Pipeline preprocessing final: alignment + equalization, lalu gabungkan
    dua jenis fitur:
      1. Piksel mentah ternormalisasi (menangkap bentuk/struktur wajah)
      2. Histogram LBP (menangkap tekstur, tahan terhadap variasi cahaya)
    Hasil akhir adalah satu vektor 1D gabungan keduanya.

    Catatan: fungsi ini dipertahankan untuk kompatibilitas pemanggil lama.
    Untuk Modular PCA, gunakan get_aligned_face() langsung lalu split_into_modules().
    """
    aligned = get_aligned_face(img, use_face_detection=use_face_detection)

    pixel_features = aligned.astype(np.float64).flatten() / 255.0
    lbp_features = lbp_histogram_features(aligned)

    return np.concatenate([pixel_features, lbp_features])


# ─────────────────────────────────────────────
# MODULAR PCA
# ─────────────────────────────────────────────
#
# Landasan teori: PCA konvensional menerapkan satu transformasi global ke
# seluruh wajah, sehingga variasi besar pada satu area (mis. mulut saat
# tersenyum lebar) ikut mendistorsi representasi area lain yang sebenarnya
# tidak berubah (mis. bentuk hidung, jarak mata). Modular PCA (Gottumukkal &
# Asari, 2004) membagi wajah jadi beberapa sub-region/modul, lalu menerapkan
# PCA terpisah pada tiap modul. Fitur lokal yang stabil antar pose/ekspresi
# (mis. area mata, hidung) tetap berkontribusi penuh ke similarity score,
# sementara distorsi di satu modul (mis. mulut) tidak "membanjiri" modul lain.
#
# Referensi: Gottumukkal, R., & Asari, V. K. (2004). An improved face
# recognition technique based on modular PCA approach. Pattern Recognition
# Letters, 25(4), 429-436.

MODULE_GRID = (3, 3)  # wajah dibagi 3x3 = 9 modul/sub-region


def split_into_modules(aligned_face: np.ndarray, grid=MODULE_GRID) -> list:
    """
    Bagi gambar wajah (2D grayscale, sudah di-align & equalize) jadi
    grid sub-region. Tiap sub-region akan diberi PCA terpisah.
    Return list of 1D arrays (tiap elemen = 1 modul, sudah di-flatten).
    """
    h, w = aligned_face.shape
    gh, gw = grid
    cell_h, cell_w = h // gh, w // gw

    modules = []
    for i in range(gh):
        for j in range(gw):
            cell = aligned_face[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            modules.append(cell.astype(np.float64).flatten() / 255.0)
    return modules


# ─────────────────────────────────────────────
# FITUR TAMBAHAN: HOG (bentuk/tepi) & SSIM (struktur piksel)
# ─────────────────────────────────────────────
#
# Landasan teori:
# - HOG (Histogram of Oriented Gradients; Dalal & Triggs, 2005): menangkap
#   distribusi arah tepi/kontur lokal pada gambar. Untuk wajah, ini menangkap
#   bentuk garis wajah (alis, rahang, hidung) secara lebih eksplisit
#   dibanding piksel mentah atau tekstur LBP, dan relatif tahan terhadap
#   variasi pencahayaan karena berbasis gradien (selisih), bukan nilai
#   piksel absolut.
# - SSIM (Structural Similarity Index; Wang et al., 2004): mengukur
#   kemiripan struktural dua gambar secara langsung (luminance, contrast,
#   structure), dipakai luas sebagai metrik kemiripan citra. Berbeda dari
#   fitur berbasis PCA, SSIM dihitung langsung antara dua gambar yang
#   dibandingkan -- jadi tidak melalui proses training, melainkan sinyal
#   tambahan independen di tahap perbandingan akhir.

@st.cache_resource(show_spinner=False)
def get_hog_descriptor():
    """HOGDescriptor OpenCV, dikonfigurasi untuk ukuran wajah ter-align (IMG_SIZE)."""
    win_size = IMG_SIZE
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    n_bins = 9
    return cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)


def hog_features(aligned_face: np.ndarray) -> np.ndarray:
    """Ekstrak fitur HOG dari wajah ter-align (2D grayscale, ukuran IMG_SIZE)."""
    hog = get_hog_descriptor()
    face_uint8 = aligned_face.astype(np.uint8)
    descriptor = hog.compute(face_uint8)
    return descriptor.flatten().astype(np.float64)


def compute_ssim(aligned_face1: np.ndarray, aligned_face2: np.ndarray) -> float:
    """
    Hitung SSIM langsung antara dua wajah ter-align. Return skor 0-1
    (1 = identik secara struktural). Dihitung di tahap compare, bukan
    lewat PCA, karena SSIM secara definisi adalah perbandingan dua
    gambar, bukan representasi satu gambar yang bisa di-training.
    """
    score = structural_similarity(
        aligned_face1.astype(np.float64),
        aligned_face2.astype(np.float64),
        data_range=255.0,
    )
    return float(score)


def extract_modular_features(img: np.ndarray, use_face_detection=True):
    """
    Pipeline Modular PCA: alignment+equalization -> bagi grid -> tiap modul
    digabung dengan fitur LBP (tekstur) dan HOG (bentuk/tepi) global sebagai
    konteks tambahan. Return: (list_modul_piksel, vektor_lbp, vektor_hog,
    gambar_wajah_teralign_untuk_SSIM)
    """
    aligned = get_aligned_face(img, use_face_detection=use_face_detection)
    modules = split_into_modules(aligned)
    lbp_feat = lbp_histogram_features(aligned)
    hog_feat = hog_features(aligned)
    return modules, lbp_feat, hog_feat, aligned


def get_face_crop(img: np.ndarray) -> np.ndarray:
    """Crop wajah untuk preview (return BGR array, bukan base64)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_box = detect_face_box(gray)

    if face_box is None:
        return cv2.resize(img, (200, 200))

    x, y, w, h = face_box
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
    """
    Buat dataset sintetis: setiap orang punya 'wajah dasar' unik + variasi noise.
    Output disusun dalam format modular (sama seperti load_dataset_from_archive)
    supaya kompatibel langsung dengan run_training() versi Modular PCA.
    """
    if log_fn: log_fn("Membuat dataset sintetis...")
    all_modules, all_lbp, all_hog, labels = [], [], [], []
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
            face = np.clip(base_face + noise, 0, 255)  # masih 0-255, belum dinormalisasi

            modules = split_into_modules(face)
            lbp = lbp_histogram_features(face.astype(np.uint8))
            hog = hog_features(face)

            all_modules.append(modules)
            all_lbp.append(lbp)
            all_hog.append(hog)
            labels.append(name)

        if log_fn: log_fn(f"  ✓ {name}: {n_images} gambar dibuat")

    n_modules = len(all_modules[0])
    modules_by_position = [
        np.array([photo_modules[m] for photo_modules in all_modules], dtype=np.float64)
        for m in range(n_modules)
    ]
    lbp_arr = np.array(all_lbp, dtype=np.float64)
    hog_arr = np.array(all_hog, dtype=np.float64)
    labels_arr = np.array(labels)

    return modules_by_position, lbp_arr, hog_arr, labels_arr


# ─────────────────────────────────────────────
# DATASET DARI ZIP UPLOAD
# ─────────────────────────────────────────────

def unwrap_single_folder(extract_dir: str) -> str:
    """
    Tembus folder pembungkus tunggal secara rekursif.
    Contoh: ZIP berisi 'Extracted/Andi/foto.jpg', 'Extracted/Budi/foto.jpg'
    -> root sebenarnya untuk dataset adalah 'Extracted/', bukan extract_dir itu sendiri.

    Berhenti begitu root punya >1 item, atau item-nya bukan folder tunggal lagi,
    atau root sudah berisi campuran folder+gambar (berarti sudah level orang).
    """
    current = extract_dir
    while True:
        items = [
            i for i in os.listdir(current)
            if not i.startswith(".") and i != "__MACOSX"
        ]
        if len(items) != 1:
            break
        only_item = os.path.join(current, items[0])
        if not os.path.isdir(only_item):
            break
        current = only_item
    return current


def detect_zip_structure(root_dir: str) -> str:
    """
    Deteksi struktur dataset di dalam root_dir (setelah folder pembungkus ditembus).
    Return 'folder' jika gambar tersusun dalam subfolder per orang,
    atau 'flat' jika semua gambar rata di root (atau campur tanpa subfolder konsisten).
    """
    root_items = os.listdir(root_dir)
    # Lewati folder sampah macOS / hidden
    root_items = [i for i in root_items if not i.startswith(".") and i != "__MACOSX"]

    subfolders = [i for i in root_items if os.path.isdir(os.path.join(root_dir, i))]
    root_images = [
        i for i in root_items
        if os.path.isfile(os.path.join(root_dir, i))
        and os.path.splitext(i)[1].lower() in ALLOWED_EXT
    ]

    # Jika ada subfolder dan masing-masing berisi gambar -> struktur folder per orang
    if len(subfolders) >= 1:
        for sub in subfolders:
            sub_path = os.path.join(root_dir, sub)
            has_image = any(
                os.path.splitext(f)[1].lower() in ALLOWED_EXT
                for f in os.listdir(sub_path)
                if os.path.isfile(os.path.join(sub_path, f))
            )
            if has_image:
                return "folder"

    # Jika ada gambar langsung di root tanpa subfolder -> flat
    if len(root_images) > 0:
        return "flat"

    # Fallback: kalau subfolder ada tapi kosong, tetap coba treat sebagai folder
    # (mungkin gambar ada di nested subfolder lagi)
    if len(subfolders) >= 1:
        return "folder"

    return "flat"


def load_dataset_from_archive(archive_bytes: bytes, filename: str, log_fn=None, progress_fn=None):
    """
    Ekstrak file arsip (.zip atau .7z, dari bytes di memori), lalu baca dataset
    dengan dua mode yang dideteksi otomatis:

    1. MODE FOLDER  -> arsip/nama_orang/foto1.jpg, foto2.jpg, ...
       Nama folder langsung dipakai sebagai label. Nama file di dalam folder bebas.

    2. MODE FLAT    -> arsip/namaorang_nomor.jpg (semua rata, tanpa folder)
       Label diambil dari nama file pakai extract_label_from_filename().

    Dioptimalkan untuk hemat memori (penting di server dgn RAM terbatas
    spt Streamlit Community Cloud free tier):
    - archive_bytes dilepas dari RAM begitu selesai diekstrak ke disk
    - Tiap foto dibaca, diproses jadi vektor fitur, lalu file gambarnya
      langsung "dilupakan" (tidak ada array gambar mentah yang menumpuk)
    - Vektor fitur dikumpulkan di list Python biasa (ringan per-elemen)
      lalu dikonversi ke satu np.array di akhir
    """
    ext = os.path.splitext(filename)[1].lower()

    if log_fn: log_fn(f"Mengekstrak dataset dari {ext.upper().lstrip('.')}...")

    extract_dir = tempfile.mkdtemp(prefix="faceds_")
    try:
        if ext == ".zip":
            with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
                zf.extractall(extract_dir)

        elif ext == ".7z":
            if not PY7ZR_AVAILABLE:
                raise ValueError(
                    "Library py7zr belum terpasang di server. "
                    "Tambahkan 'py7zr' ke requirements.txt lalu reboot app."
                )
            with py7zr.SevenZipFile(io.BytesIO(archive_bytes), mode="r") as sz:
                sz.extractall(path=extract_dir)

        else:
            raise ValueError(f"Format file '{ext}' tidak didukung. Gunakan .zip atau .7z")

        # Lepas bytes arsip mentah dari memori sesegera mungkin -- begitu
        # sudah diekstrak ke disk, kita tidak butuh salinan di RAM lagi.
        # Untuk dataset 200MB+, ini langsung membebaskan 200MB+ RAM.
        del archive_bytes
        import gc
        gc.collect()

        # Tembus folder pembungkus tunggal, mis. "Extracted/" yang isinya
        # langsung folder-per-orang, supaya tidak salah dianggap "1 orang".
        root_dir = unwrap_single_folder(extract_dir)
        if log_fn and root_dir != extract_dir:
            wrapper_name = os.path.relpath(root_dir, extract_dir)
            log_fn(f"  Folder pembungkus terdeteksi & dilewati: {wrapper_name}/")

        structure = detect_zip_structure(root_dir)
        if log_fn:
            mode_label = "Folder per orang" if structure == "folder" else "File rata (flat)"
            log_fn(f"  Struktur terdeteksi: {mode_label}")

        # Kumpulkan dulu daftar (filepath, label) tanpa membuka gambarnya --
        # ini ringan walau jumlah file ribuan, karena cuma teks path.
        file_label_pairs = []

        if structure == "folder":
            root_items = sorted(os.listdir(root_dir))
            person_folders = [
                i for i in root_items
                if os.path.isdir(os.path.join(root_dir, i))
                and not i.startswith(".") and i != "__MACOSX"
            ]
            for person_name in person_folders:
                label = person_name.strip().lower().replace(" ", "_")
                person_path = os.path.join(root_dir, person_name)
                for root, _, files in os.walk(person_path):
                    for fname in sorted(files):
                        fext = os.path.splitext(fname)[1].lower()
                        if fext not in ALLOWED_EXT or fname.startswith("."):
                            continue
                        file_label_pairs.append((os.path.join(root, fname), label))

        else:  # flat
            for root, _, files in os.walk(root_dir):
                for fname in sorted(files):
                    fext = os.path.splitext(fname)[1].lower()
                    if fext not in ALLOWED_EXT or fname.startswith("._") or fname.startswith("."):
                        continue
                    label = extract_label_from_filename(fname)
                    file_label_pairs.append((os.path.join(root, fname), label))

        total_files = len(file_label_pairs)
        if log_fn:
            log_fn(f"  {total_files} file gambar ditemukan, memproses satu per satu...")

        # Proses satu per satu: baca -> ekstrak fitur modular -> buang gambar
        # mentahnya segera (img dan vec sementara, tidak ada penumpukan array besar).
        # Catatan: gambar 'aligned' dari extract_modular_features TIDAK disimpan
        # di sini (cuma dipakai untuk SSIM saat compare 2 foto, bukan saat training
        # banyak foto sekaligus -- supaya tidak boros memori untuk dataset besar).
        all_modules, all_lbp, all_hog, labels = [], [], [], []
        skipped = 0
        report_every = max(1, total_files // 10)  # update progress tiap ~10%

        for idx, (fpath, label) in enumerate(file_label_pairs):
            img = cv2.imread(fpath)
            if img is None:
                skipped += 1
                continue
            try:
                modules, lbp, hog, _aligned = extract_modular_features(img, use_face_detection=True)
            except Exception:
                skipped += 1
                continue
            finally:
                del img  # lepas gambar mentah dari memori segera setelah dipakai

            all_modules.append(modules)
            all_lbp.append(lbp)
            all_hog.append(hog)
            labels.append(label)

            if progress_fn and (idx + 1) % report_every == 0:
                progress_fn(idx + 1, total_files)
            if (idx + 1) % 200 == 0:
                gc.collect()  # bersihkan memori berkala saat dataset besar

        if progress_fn:
            progress_fn(total_files, total_files)

        if len(all_modules) == 0:
            raise ValueError(
                "Tidak ada gambar valid ditemukan di dalam arsip. "
                "Pastikan struktur folder per orang (nama_orang/foto.jpg) "
                "atau nama file flat (nama_orang_nomor.jpg) sudah benar."
            )

        unique_labels = sorted(set(labels))
        if log_fn:
            log_fn(f"  ✓ {len(all_modules)} gambar berhasil dibaca, {len(unique_labels)} orang terdeteksi")
            if skipped > 0:
                log_fn(f"  ⚠ {skipped} file dilewati (gagal dibaca / wajah tidak terdeteksi)")
            for lbl in unique_labels:
                count = labels.count(lbl)
                log_fn(f"    - {lbl}: {count} foto")

        # Susun ulang per-modul: dari list (per foto) of list (per modul)
        # menjadi list (per modul) of array (per foto) -> bentuk yang
        # dibutuhkan untuk melatih satu PCA per modul di run_training().
        n_modules = len(all_modules[0])
        modules_by_position = [
            np.array([photo_modules[m] for photo_modules in all_modules], dtype=np.float64)
            for m in range(n_modules)
        ]
        lbp_arr = np.array(all_lbp, dtype=np.float64)
        hog_arr = np.array(all_hog, dtype=np.float64)
        labels_arr = np.array(labels)

        del all_modules, all_lbp, all_hog, labels
        gc.collect()

        return modules_by_position, lbp_arr, hog_arr, labels_arr

    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


# ─────────────────────────────────────────────
# TRAINING PCA
# ─────────────────────────────────────────────

def run_training(modules_by_position, lbp_arr: np.ndarray, hog_arr: np.ndarray,
                  labels: np.ndarray, log_fn=None):
    """
    Training Modular PCA: satu PCA terpisah dilatih untuk tiap modul/sub-region
    wajah (mis. 9 modul untuk grid 3x3), plus satu PCA untuk fitur LBP global
    (tekstur) dan satu PCA untuk fitur HOG global (bentuk/tepi).

    Landasan teori (Gottumukkal & Asari, 2004): membagi wajah jadi sub-region
    membuat variasi lokal (ekspresi di area mulut, dsb.) tidak mendistorsi
    representasi area lain yang stabil (mata, hidung). Setiap modul berkontribusi
    independen ke skor similarity akhir.

    Eigenface discarding (Turk & Pentland, 1991 // dianjurkan literatur lanjutan):
    komponen utama pertama dari PCA wajah biasanya didominasi oleh variasi
    pencahayaan global, bukan informasi identitas. N_EIGENFACE_DISCARD komponen
    pertama dari tiap PCA modul dibuang sebelum dipakai untuk similarity.

    HOG (Dalal & Triggs, 2005): fitur gradien tepi/kontur, menangkap bentuk
    wajah secara eksplisit, dilatih sebagai PCA terpisah seperti LBP.
    """
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        raise ValueError("Minimal perlu 2 orang berbeda dalam dataset untuk training")
    if len(labels) < 3:
        raise ValueError("Minimal perlu 3 foto total dalam dataset untuk training")

    n_samples = len(labels)
    n_modules = len(modules_by_position)
    if log_fn:
        log_fn(f"\nDataset: {n_samples} gambar | {n_unique} orang")
        log_fn(f"Modular PCA: wajah dibagi {MODULE_GRID[0]}x{MODULE_GRID[1]} = {n_modules} modul")

    t_start = time.time()

    # Komponen per modul disesuaikan dengan ukuran modul itu sendiri & jumlah
    # sampel, supaya tidak minta lebih komponen daripada yang tersedia.
    module_pcas = []
    module_scalers = []
    X_pca_modules = []

    for m_idx, X_module in enumerate(modules_by_position):
        scaler_m = StandardScaler()
        X_module_scaled = scaler_m.fit_transform(X_module)

        max_comp = min(N_COMPONENTS_PER_MODULE, n_samples - 1, X_module.shape[1])
        # Sisakan ruang untuk discard eigenface pertama -- minimal 1 komponen tersisa
        max_comp = max(max_comp, N_EIGENFACE_DISCARD + 1)

        pca_m = PCA(n_components=max_comp, svd_solver="randomized", random_state=42)
        Z_m = pca_m.fit_transform(X_module_scaled)

        module_pcas.append(pca_m)
        module_scalers.append(scaler_m)
        X_pca_modules.append(Z_m)

    # PCA terpisah untuk fitur LBP global (tekstur, tahan-cahaya)
    lbp_scaler = StandardScaler()
    lbp_scaled = lbp_scaler.fit_transform(lbp_arr)
    lbp_n_comp = min(N_COMPONENTS_LBP, n_samples - 1, lbp_arr.shape[1])
    lbp_pca = PCA(n_components=lbp_n_comp, svd_solver="randomized", random_state=42)
    X_pca_lbp = lbp_pca.fit_transform(lbp_scaled)

    # PCA terpisah untuk fitur HOG global (bentuk/tepi)
    hog_scaler = StandardScaler()
    hog_scaled = hog_scaler.fit_transform(hog_arr)
    hog_n_comp = min(N_COMPONENTS_HOG, n_samples - 1, hog_arr.shape[1])
    hog_pca = PCA(n_components=hog_n_comp, svd_solver="randomized", random_state=42)
    X_pca_hog = hog_pca.fit_transform(hog_scaled)

    t_elapsed = time.time() - t_start

    avg_explained = float(np.mean([
        np.sum(p.explained_variance_ratio_) for p in module_pcas
    ])) * 100

    if log_fn:
        log_fn(f"  ✓ {n_modules} PCA modul + PCA LBP + PCA HOG selesai dalam {t_elapsed:.1f} detik")
        log_fn(f"  ✓ {N_EIGENFACE_DISCARD} eigenface pertama tiap modul dibuang (dominasi pencahayaan)")
        log_fn(f"  ✓ Rata-rata explained variance per modul: {avg_explained:.1f}%")

    # Kalibrasi skala jarak euclidean gabungan dari data training itu sendiri.
    # Pakai representasi gabungan semua modul (setelah discard) + LBP + HOG
    # untuk estimasi skala yang representatif terhadap skor compare_faces().
    sample_n = min(300, n_samples)
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(n_samples, size=sample_n, replace=False)

    combined_train_vectors = []
    for i in idx_sample:
        parts = [Z[i, N_EIGENFACE_DISCARD:] for Z in X_pca_modules]
        parts.append(X_pca_lbp[i])
        parts.append(X_pca_hog[i])
        combined_train_vectors.append(np.concatenate(parts))
    combined_train_vectors = np.array(combined_train_vectors)

    sample_dists = euclidean_distances(combined_train_vectors)
    upper_tri = sample_dists[np.triu_indices(sample_n, k=1)]
    dist_scale = float(np.percentile(upper_tri, 90)) if len(upper_tri) > 0 else 30.0
    dist_scale = max(dist_scale, 1e-6)

    if log_fn:
        log_fn(f"  ✓ Skala jarak euclidean terkalibrasi: {dist_scale:.2f}")
        log_fn(f"\n✅ Model berhasil ditraining! Siap membandingkan wajah.")

    return {
        "module_pcas": module_pcas,
        "module_scalers": module_scalers,
        "lbp_pca": lbp_pca,
        "lbp_scaler": lbp_scaler,
        "hog_pca": hog_pca,
        "hog_scaler": hog_scaler,
        "labels": labels,
        "dist_scale": dist_scale,
        "n_modules": n_modules,
    }


def project_image_to_model(model: dict, img: np.ndarray):
    """
    Proyeksikan satu gambar ke ruang Modular PCA model: ekstrak modul + LBP
    + HOG, transform tiap modul dengan PCA-nya masing-masing (buang
    N_EIGENFACE_DISCARD komponen pertama), gabung jadi satu vektor representasi
    akhir. Juga mengembalikan gambar wajah ter-align (dibutuhkan untuk SSIM,
    yang dihitung langsung di compare_faces, bukan lewat PCA).
    """
    modules, lbp, hog, aligned = extract_modular_features(img)

    parts = []
    for m_idx, module_vec in enumerate(modules):
        scaler_m = model["module_scalers"][m_idx]
        pca_m = model["module_pcas"][m_idx]
        scaled = scaler_m.transform(module_vec.reshape(1, -1))
        projected = pca_m.transform(scaled)[0]
        parts.append(projected[N_EIGENFACE_DISCARD:])

    lbp_scaled = model["lbp_scaler"].transform(lbp.reshape(1, -1))
    lbp_projected = model["lbp_pca"].transform(lbp_scaled)[0]
    parts.append(lbp_projected)

    hog_scaled = model["hog_scaler"].transform(hog.reshape(1, -1))
    hog_projected = model["hog_pca"].transform(hog_scaled)[0]
    parts.append(hog_projected)

    return np.concatenate(parts), aligned


def compare_faces(model: dict, img1: np.ndarray, img2: np.ndarray,
                   weight_cosine: float = 0.45, weight_euclidean: float = 0.30,
                   weight_ssim: float = 0.25) -> dict:
    """
    Bandingkan dua wajah. Skor akhir adalah weighted sum dari 3 sinyal:
    - Cosine similarity di ruang Modular PCA (gabungan modul + LBP + HOG)
    - Euclidean similarity (1 - jarak ternormalisasi) di ruang yang sama
    - SSIM: kemiripan struktural piksel langsung antara 2 wajah ter-align

    Bobot bisa diatur (weight_cosine, weight_euclidean, weight_ssim) -- akan
    dinormalisasi otomatis supaya totalnya selalu 1.0 berapapun nilai mentahnya.
    """
    z1, aligned1 = project_image_to_model(model, img1)
    z2, aligned2 = project_image_to_model(model, img2)
    z1, z2 = z1.reshape(1, -1), z2.reshape(1, -1)

    cos_sim = float(cosine_similarity(z1, z2)[0][0])
    euc_dist = float(euclidean_distances(z1, z2)[0][0])

    dist_scale = model.get("dist_scale", 30.0)
    euc_sim = max(0.0, 1.0 - min(euc_dist / dist_scale, 1.0))

    ssim_sim = compute_ssim(aligned1, aligned2)
    ssim_sim = max(0.0, ssim_sim)  # SSIM bisa negatif untuk citra sangat berbeda; clamp ke 0

    # Normalisasi bobot supaya totalnya selalu 1.0 (robust terhadap input
    # sembarang dari slider UI, mis. kalau user set semua ke 0).
    total_w = weight_cosine + weight_euclidean + weight_ssim
    if total_w <= 0:
        weight_cosine, weight_euclidean, weight_ssim = 0.45, 0.30, 0.25
        total_w = 1.0
    wc, we, ws = weight_cosine / total_w, weight_euclidean / total_w, weight_ssim / total_w

    combined = cos_sim * wc + euc_sim * we + ssim_sim * ws

    if combined >= 0.50:
        verdict, sub, level, emoji = "MIRIP", "Kemungkinan besar orang yang sama", "high", "✅"
    elif combined >= 0.40:
        verdict, sub, level, emoji = "CUKUP MIRIP", "Mungkin orang yang sama", "medium", "🤔"
    else:
        verdict, sub, level, emoji = "TIDAK MIRIP", "Kemungkinan besar orang berbeda", "none", "❌"

    return {
        "cosine_similarity": round(cos_sim * 100, 1),
        "euclidean_sim": round(euc_sim * 100, 1),
        "ssim_similarity": round(ssim_sim * 100, 1),
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
if "auto_load_attempted" not in st.session_state:
    st.session_state.auto_load_attempted = False
if "auto_load_error" not in st.session_state:
    st.session_state.auto_load_error = None


def log(msg):
    st.session_state.train_log.append(msg)


def try_auto_load_model():
    """
    Kalau ada URL model dikonfigurasi lewat Streamlit secrets
    (key: MODEL_URL, mis. raw link GitHub ke model.pkl), coba load
    otomatis sekali saat app pertama kali dibuka -- supaya pengguna
    tidak perlu klik manual tiap kali, sesuai kebutuhan "akses otomatis".

    Konfigurasi di Streamlit Cloud: Settings -> Secrets, isi:
        MODEL_URL = "https://raw.githubusercontent.com/user/repo/main/model.pkl"
    """
    if st.session_state.auto_load_attempted or st.session_state.model is not None:
        return
    st.session_state.auto_load_attempted = True

    model_url = st.secrets.get("MODEL_URL", "") if hasattr(st, "secrets") else ""
    if not model_url:
        return

    try:
        resp = requests.get(model_url, timeout=30)
        resp.raise_for_status()
        model = deserialize_model(resp.content)
        st.session_state.model = model
        st.session_state.model_url = model_url
    except Exception as e:
        st.session_state.auto_load_error = str(e)


def serialize_model(model: dict) -> bytes:
    """Konversi model hasil training jadi bytes pickle, siap didownload / diupload ke GitHub."""
    import pickle
    return pickle.dumps(model)


def deserialize_model(data: bytes) -> dict:
    """
    Baca bytes pickle jadi dict model. Validasi struktur minimal supaya
    error-nya jelas kalau file yang diupload bukan model yang valid/cocok.
    """
    import pickle
    try:
        model = pickle.loads(data)
    except Exception as e:
        raise ValueError(f"File tidak bisa dibaca sebagai model (.pkl) yang valid: {e}")

    required_keys = {"module_pcas", "module_scalers", "lbp_pca", "lbp_scaler", "labels"}
    if not isinstance(model, dict) or not required_keys.issubset(model.keys()):
        raise ValueError(
            "Struktur file tidak sesuai format model aplikasi ini. "
            "Pastikan file .pkl berasal dari hasil training/download di app ini."
        )
    return model


# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────

st.markdown("## 🧠 Face Match ML")
st.caption("Training model PCA sendiri, lalu bandingkan dua wajah — apakah orang yang sama atau bukan.")

try_auto_load_model()
if st.session_state.model is not None and st.session_state.get("model_url"):
    st.success(f"✅ Model otomatis dimuat dari URL terkonfigurasi.", icon="📥")
elif st.session_state.auto_load_error:
    st.warning(
        f"⚠️ Gagal memuat model otomatis dari MODEL_URL: {st.session_state.auto_load_error}. "
        "Bisa training manual atau load manual di tab terkait.",
        icon="⚠️",
    )

tab_train, tab_load, tab_compare = st.tabs([
    "⚙️ 1. Training Model", "📥 2. Load Model Siap Pakai", "🔍 3. Bandingkan Wajah"
])


# ─────────────────────────────────────────────
# TAB 1 — TRAINING
# ─────────────────────────────────────────────

with tab_train:

    if st.session_state.model is not None:
        m = st.session_state.model
        n_persons = len(np.unique(m["labels"]))
        n_samples = len(m["labels"])
        avg_explained = float(np.mean([
            np.sum(p.explained_variance_ratio_) for p in m["module_pcas"]
        ])) * 100
        st.success(
            f"✅ **Model sudah tertraining!**  \n"
            f"{n_persons} orang · {n_samples} sampel · {m['n_modules']} modul PCA "
            f"({MODULE_GRID[0]}x{MODULE_GRID[1]}) + 1 PCA LBP · "
            f"{avg_explained:.1f}% rata-rata explained variance"
        )

        model_bytes = serialize_model(m)
        st.download_button(
            "💾 Download Model (.pkl)",
            data=model_bytes,
            file_name="model.pkl",
            mime="application/octet-stream",
            use_container_width=True,
            help=(
                "Simpan file ini, lalu upload ke repo GitHub kamu. "
                "Setelah itu bisa langsung di-load di tab 'Load Model Siap Pakai' "
                "tanpa perlu training ulang."
            ),
        )
        st.caption(
            f"📦 Ukuran file: {len(model_bytes) / 1024 / 1024:.1f} MB — "
            "GitHub batasi file biasa maks 100MB per file (di luar Git LFS)."
        )

    method = st.radio(
        "Pilih metode dataset",
        ["🎲 Dataset Sintetis", "📦 Upload Arsip (ZIP/7z)"],
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
                    modules, lbp_arr, hog_arr, labels = create_synthetic_dataset(n_persons, n_images, log_fn=log)
                    log_box.code("\n".join(st.session_state.train_log))
                    model = run_training(modules, lbp_arr, hog_arr, labels, log_fn=log)
                    log_box.code("\n".join(st.session_state.train_log))
                st.session_state.model = model
                st.rerun()
            except Exception as e:
                log(f"\n❌ Error: {e}")
                log_box.code("\n".join(st.session_state.train_log))
                st.error(str(e))

    else:  # Upload Arsip
        st.markdown("""
        <div class="format-hint">
        📋 <strong>Format arsip yang didukung:</strong> <code>.zip</code> dan <code>.7z</code><br><br>
        <strong>Struktur isi (otomatis terdeteksi):</strong><br><br>
        <strong>1. Folder per orang</strong> (direkomendasikan untuk dataset besar)<br>
        <code>dataset/andi/foto1.jpg</code>, <code>dataset/andi/foto2.jpg</code><br>
        <code>dataset/budi/apa_saja.png</code> — nama folder = nama orang<br><br>
        <strong>2. File rata (flat)</strong><br>
        <code>dataset/andi_1.jpg</code>, <code>dataset/budi_1.png</code><br>
        — nama orang diambil dari nama file sebelum angka terakhir<br><br>
        Minimal 2 orang berbeda. Format gambar: jpg, jpeg, png, bmp, webp
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        archive_file = st.file_uploader("Upload file dataset (.zip atau .7z)", type=["zip", "7z"])

        if st.button("🚀 Mulai Training (dari Arsip)", type="primary",
                      use_container_width=True, disabled=(archive_file is None)):
            st.session_state.train_log = []
            log_box = st.empty()
            progress_bar = st.progress(0, text="Menyiapkan...")
            try:
                archive_bytes = archive_file.read()

                def update_progress(done, total):
                    pct = done / total if total > 0 else 0
                    progress_bar.progress(pct, text=f"Memproses foto {done}/{total}...")

                modules, lbp_arr, hog_arr, labels = load_dataset_from_archive(
                    archive_bytes, archive_file.name,
                    log_fn=log, progress_fn=update_progress,
                )
                log_box.code("\n".join(st.session_state.train_log))
                progress_bar.progress(1.0, text="Memproses foto selesai, melatih PCA...")

                with st.spinner("Training PCA..."):
                    model = run_training(modules, lbp_arr, hog_arr, labels, log_fn=log)
                log_box.code("\n".join(st.session_state.train_log))

                progress_bar.empty()
                st.session_state.model = model
                st.rerun()
            except Exception as e:
                progress_bar.empty()
                log(f"\n❌ Error: {e}")
                log_box.code("\n".join(st.session_state.train_log))
                st.error(
                    f"{e}\n\n"
                    "Catatan: jika ini terjadi pada dataset besar (ratusan MB), "
                    "kemungkinan penyebabnya server kehabisan memori (RAM). "
                    "Coba kurangi jumlah foto / orang per batch training."
                )

    if st.session_state.train_log and st.session_state.model is None:
        with st.expander("📜 Log training terakhir", expanded=True):
            st.code("\n".join(st.session_state.train_log))


# ─────────────────────────────────────────────
# TAB 2 — LOAD MODEL SIAP PAKAI
# ─────────────────────────────────────────────

with tab_load:
    st.caption(
        "Sudah punya model hasil training sebelumnya (file `.pkl`)? Load di sini supaya "
        "tidak perlu training ulang setiap kali app dibuka. Bisa dari URL (mis. GitHub) "
        "atau upload file langsung."
    )

    if st.session_state.model is not None:
        m = st.session_state.model
        st.info(
            f"ℹ️ Saat ini ada model aktif: {len(np.unique(m['labels']))} orang, "
            f"{len(m['labels'])} sampel. Load model baru akan menggantikannya."
        )

    load_method = st.radio(
        "Sumber model",
        ["🔗 Dari URL (mis. GitHub raw link)", "📁 Upload file .pkl"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.write("")

    if load_method == "🔗 Dari URL (mis. GitHub raw link)":
        st.markdown("""
        <div class="format-hint">
        📋 <strong>Cara dapatkan link raw GitHub:</strong><br>
        1. Upload file <code>model.pkl</code> ke repo GitHub kamu (di luar folder yang di-ignore)<br>
        2. Buka file itu di GitHub, klik tombol <strong>"Raw"</strong><br>
        3. Copy URL dari address bar (formatnya seperti
        <code>https://raw.githubusercontent.com/user/repo/main/model.pkl</code>)<br>
        4. Tempel di kolom bawah ini
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        default_url = st.session_state.get("model_url", "")
        model_url = st.text_input(
            "URL model.pkl",
            value=default_url,
            placeholder="https://raw.githubusercontent.com/username/repo/main/model.pkl",
        )

        if st.button("📥 Load Model dari URL", type="primary",
                      use_container_width=True, disabled=(not model_url.strip())):
            try:
                with st.spinner("Mengunduh dan memuat model..."):
                    resp = requests.get(model_url.strip(), timeout=30)
                    resp.raise_for_status()
                    model = deserialize_model(resp.content)
                st.session_state.model = model
                st.session_state.model_url = model_url.strip()
                st.success("✅ Model berhasil dimuat dari URL!")
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Gagal mengunduh file dari URL: {e}")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Terjadi kesalahan tak terduga: {e}")

    else:  # Upload file .pkl
        pkl_file = st.file_uploader("Upload file model (.pkl)", type=["pkl"])

        if st.button("📥 Load Model dari File", type="primary",
                      use_container_width=True, disabled=(pkl_file is None)):
            try:
                with st.spinner("Memuat model..."):
                    model = deserialize_model(pkl_file.read())
                st.session_state.model = model
                st.success("✅ Model berhasil dimuat!")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Terjadi kesalahan tak terduga: {e}")


# ─────────────────────────────────────────────
# TAB 3 — COMPARE
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

        if st.session_state.get("__reset_weights"):
            st.session_state["w_cosine"] = 0.45
            st.session_state["w_euclidean"] = 0.30
            st.session_state["w_ssim"] = 0.25
            st.session_state["__reset_weights"] = False

        with st.expander("⚙️ Bobot Skor (lanjutan)", expanded=False):
            st.caption(
                "Atur kontribusi tiap sinyal similarity ke skor akhir. "
                "Nilai dinormalisasi otomatis (jumlah tidak harus 1.0)."
            )
            wc1, wc2, wc3 = st.columns(3)
            with wc1:
                weight_cosine = st.slider("Cosine (Modular PCA)", 0.0, 1.0, 0.45, 0.05, key="w_cosine")
            with wc2:
                weight_euclidean = st.slider("Euclidean (Modular PCA)", 0.0, 1.0, 0.30, 0.05, key="w_euclidean")
            with wc3:
                weight_ssim = st.slider("SSIM (Struktur Piksel)", 0.0, 1.0, 0.25, 0.05, key="w_ssim")

            total_weight_check = weight_cosine + weight_euclidean + weight_ssim
            if total_weight_check <= 0.01:
                st.warning(
                    "⚠️ Semua bobot di angka 0 (atau mendekati 0) — skor akhir tidak akan "
                    "merepresentasikan similarity yang sebenarnya. Geser minimal satu slider "
                    "ke nilai > 0, atau gunakan tombol reset di bawah.",
                    icon="⚠️",
                )
            if st.button("↺ Reset Bobot ke Default (0.45 / 0.30 / 0.25)"):
                st.session_state["__reset_weights"] = True
                st.rerun()

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

                    result = compare_faces(
                        st.session_state.model, img1, img2,
                        weight_cosine=weight_cosine,
                        weight_euclidean=weight_euclidean,
                        weight_ssim=weight_ssim,
                    )

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

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Skor Gabungan", f"{result['combined_score']}%")
                s2.metric("Cosine (Modular PCA)", f"{result['cosine_similarity']}%")
                s3.metric("Euclidean (Modular PCA)", f"{result['euclidean_sim']}%")
                s4.metric("SSIM (Struktur)", f"{result['ssim_similarity']}%")

                n_persons = len(np.unique(st.session_state.model["labels"]))
                n_samples = len(st.session_state.model["labels"])
                st.caption(
                    f"⚙️ Model: {n_persons} orang, {n_samples} sampel training — "
                    f"Modular PCA + LBP + HOG + SSIM. "
                    f"Skor ≥50% mirip · 40-49% cukup mirip · <40% tidak mirip."
                )

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

        if st.button("↺ Reset Foto"):
            st.session_state.pop("up1", None)
            st.session_state.pop("up2", None)
            st.rerun()


st.markdown("---")
st.caption("Berbasis face_recognation.py · PCA/SVD Eigenfaces · Streamlit + OpenCV")
