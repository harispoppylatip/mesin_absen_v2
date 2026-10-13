"""Layanan pengenal wajah InsightFace untuk aplikasi absen solat.

Kenapa dipisah dari PHP
-----------------------
Memuat model InsightFace butuh beberapa detik dan memakan ratusan megabita
memori. Kalau dijalankan dari PHP pada setiap permintaan, presensi jadi lambat.
Modelnya dimuat SEKALI di proses ini, lalu aplikasi sekolah cukup mengirim foto
dan menerima angka wajahnya.

Kontrak balasan sengaja dibuat sama dengan layanan wajah sebelumnya (dlib),
supaya aplikasi Laravel tidak perlu diubah selain dimensi angka wajah, nama
model, dan ambangnya.

    GET  /health
    POST /encode    multipart, nama field "image"
    POST /detect    multipart, nama field "image"

Perbedaan yang WAJIB diketahui
------------------------------
1. Dimensi angka wajah 512, bukan 128. Angka model lama tidak bisa dibandingkan
   dengan angka model ini, jadi baris wajah lama memang tidak terpakai lagi.
2. Angka dari InsightFace TIDAK ternormalisasi (panjangnya sekitar 27), jadi
   layanan ini menormalkannya sebelum dikirim. Dengan panjang 1, jarak euclid
   antar dua angka setara dengan cosine similarity, sehingga angka di layar
   aplikasi tetap berarti "makin kecil makin mirip".
3. Nama model yang dilaporkan adalah `insightface-<nama model>`, dan nilai itu
   harus sama dengan FACE_MODEL di .env aplikasi Laravel. Bila berbeda, aplikasi
   menolak angkanya dengan sengaja.

Optimasi yang dipakai (semuanya diukur, bukan diandaikan)
---------------------------------------------------------
* Model dimuat sekali, dan sesudah itu hanya satu salinan di memori.
* Bentuk input detektor mengikuti rasio gambar. Pada gambar 1280x720 median
  waktu deteksi turun dari 16,0 ms (kanvas persegi 640) menjadi 8,1 ms
  (kanvas 640x352). Semua angka diukur lima kali di i7-13620H.
* Angka wajah hanya dihitung untuk wajah yang layak dinilai, sehingga model
  pengenalan tidak dijalankan untuk jawaban yang langsung dibuang.
* Jumlah thread dibatasi. Membiarkan onnxruntime memakai seluruh core terbukti
  lebih lambat pada CPU dengan P-core dan E-core bercampur.
* Jumlah wajah yang dihitung per bingkai dibatasi (FACE_MAX_FACES).
* Pemanasan dijalankan saat mulai, karena panggilan pengenalan pertama bisa
  tujuh kali lebih lambat daripada panggilan berikutnya.

Menjalankan
-----------
    python app.py                 layanan HTTP di FACE_HOST:FACE_PORT
    python app.py --preload       unduh dan periksa model lalu keluar
    python app.py --check         cetak versi pustaka dan pengaturan aktif
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

# Impor FastAPI sengaja di TINGKAT MODUL, bukan di dalam build_app().
# Berkas ini memakai `from __future__ import annotations`, sehingga seluruh
# anotasi tipe menjadi teks dan FastAPI harus mencarinya di ruang nama modul
# ini. Ketika impornya diletakkan di dalam fungsi, nama seperti UploadFile dan
# Response TIDAK ketemu, dan FastAPI diam-diam menganggapnya sebagai parameter
# kueri biasa sehingga setiap permintaan ditolak dengan 422.
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

# =====================================================================
# VARIABEL LINGKUP PROSES
# Harus diisi SEBELUM onnxruntime diimpor, sebab pustaka itu membaca
# variabel ini saat dimuat. Menyetelnya belakangan tidak berpengaruh.
# =====================================================================


def _benar(nilai: str | None, bawaan: bool = False) -> bool:
    """Baca nilai benar atau salah dari variabel lingkungan."""
    if nilai is None or str(nilai).strip() == "":
        return bawaan

    return str(nilai).strip().lower() in {"1", "true", "ya", "yes", "on"}


def _angka(nilai: str | None, bawaan: int) -> int:
    """Baca bilangan bulat dari variabel lingkungan, dengan nilai cadangan."""
    try:
        return int(str(nilai).strip())
    except (TypeError, ValueError):
        return bawaan


def _pecahan(nilai: str | None, bawaan: float) -> float:
    """Baca bilangan pecahan dari variabel lingkungan."""
    try:
        return float(str(nilai).strip())
    except (TypeError, ValueError):
        return bawaan


_DIRI = Path(__file__).resolve().parent

# Alamat dan port layanan. Bawaannya 127.0.0.1, artinya hanya menerima koneksi
# dari komputer yang sama. Layanan ini tidak punya kata sandi, jadi membukanya
# ke jaringan harus selalu keputusan sadar. Di dalam Docker, Dockerfile dan
# docker-compose.yml mengisinya 0.0.0.0 supaya kontainer lain bisa menghubungi.
HOST = os.getenv("FACE_HOST", "127.0.0.1")
PORT = _angka(os.getenv("FACE_PORT"), 5005)

# --- Model ----------------------------------------------------------
# buffalo_sc = det_500m + MobileFaceNet (w600k_mbf), jauh lebih ringan daripada
# buffalo_l. Diukur di i7-13620H satu wajah: buffalo_sc sekitar 20 ms sedangkan
# buffalo_l 168 ms.
MODEL_NAME = os.getenv("FACE_MODEL", "buffalo_sc")
MODEL_ROOT = Path(os.getenv("FACE_MODEL_ROOT", str(_DIRI / "cache" / "models")))
# Nama yang dilaporkan ke aplikasi. Harus sama dengan FACE_MODEL di .env website.
MODEL_LABEL = os.getenv("FACE_MODEL_NAME", f"insightface-{MODEL_NAME}")
# Dimensi angka wajah InsightFace. Harus sama dengan FACE_EMBEDDING_DIMENSIONS
# di .env website.
DIMENSIONS = _angka(os.getenv("FACE_DIMENSIONS"), 512)
USE_GPU = _benar(os.getenv("FACE_USE_GPU"), False)
GPU_DEVICE_ID = _angka(os.getenv("FACE_GPU_DEVICE_ID"), 0)

# Ambang keyakinan detektor. Semakin kecil, semakin banyak wajah samar diterima.
DET_THRESHOLD = _pecahan(os.getenv("FACE_DET_THRESHOLD"), 0.5)
# Sisi panjang kanvas detektor. Kecilkan ke 320 untuk CPU lemah, dengan risiko
# wajah kecil atau wajah jauh terlewat.
DETECT_LONG_SIDE = _angka(os.getenv("FACE_DETECT_LONG_SIDE"), 640)
# Bentuk kanvas mengikuti rasio gambar. Terukur hampir dua kali lebih cepat
# pada gambar 1280x720 karena kanvas tidak lagi penuh padding nol.
DETECT_MATCH_ASPECT = _benar(os.getenv("FACE_DETECT_MATCH_ASPECT"), True)

# Paling banyak wajah yang dihitung angka wajahnya dalam satu gambar. Isi 0
# untuk tidak membatasi.
MAX_FACES = _angka(os.getenv("FACE_MAX_FACES"), 8)
# Wajah yang sisi terpendeknya di bawah angka ini tidak dihitung angka wajahnya.
# Hanya pengaman CPU: potret seperti itu tidak layak diputuskan identitasnya,
# dan aplikasi menolaknya sendiri lewat FACE_MIN_FACE_SIZE.
EMBED_MIN_FACE_SIZE = _angka(os.getenv("FACE_EMBED_MIN_FACE_SIZE"), 24)
# Batas jumlah piksel gambar yang diterima. Foto 40 megapiksel masih diterima,
# di atas itu ditolak supaya satu kiriman tidak menghabiskan memori server.
MAX_IMAGE_PIXELS = _angka(os.getenv("FACE_MAX_IMAGE_PIXELS"), 40_000_000)

# --- Thread ---------------------------------------------------------
# Terlalu banyak thread JUSTRU lebih lambat pada CPU dengan P-core dan E-core
# bercampur. Angka di bawah titik aman; ukur di server dengan benchmark.py.
ORT_INTRA_THREADS = _angka(os.getenv("FACE_ORT_INTRA_THREADS"), 4)
ORT_INTER_THREADS = _angka(os.getenv("FACE_ORT_INTER_THREADS"), 1)
OPENCV_THREADS = _angka(os.getenv("FACE_OPENCV_THREADS"), 1)

# --- Foto bukti absen -----------------------------------------------
PROOF_MAX_SIDE = _angka(os.getenv("FACE_PROOF_MAX_SIDE"), 480)
PROOF_QUALITY = _angka(os.getenv("FACE_PROOF_QUALITY"), 72)
PROOF_MAX_KB = _angka(os.getenv("FACE_PROOF_MAX_KB"), 48)
PROOF_MAX_BYTES = PROOF_MAX_KB * 1024
PROOF_QUALITIES = tuple(max(30, PROOF_QUALITY - langkah * 8) for langkah in range(5))

# --- Pemanasan ------------------------------------------------------
# Panggilan pengenalan pertama bisa tujuh kali lebih lambat daripada panggilan
# berikutnya. Pemanasan saat mulai memindahkan biaya itu ke waktu tunggu
# layanan, bukan ke pengguna pertama.
WARMUP = _benar(os.getenv("FACE_WARMUP"), True)

LOG_LEVEL = os.getenv("FACE_LOG_LEVEL", "INFO").upper()

# Batasi thread OpenMP sebelum onnxruntime dimuat. Tanpa ini, thread OpenMP
# bawaan bisa jauh melebihi jumlah core dan saling berebut.
if ORT_INTRA_THREADS > 0:
    os.environ.setdefault("OMP_NUM_THREADS", str(ORT_INTRA_THREADS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("layanan-wajah-insightface")


# =====================================================================
# UTILITAS GAMBAR
# =====================================================================


def decode_image(data: bytes) -> "Any | None":
    """Baca gambar dari bita menjadi larik BGR OpenCV.

    Memakai cv2.imdecode dari larik, bukan cv2.imread, supaya jalur berkas
    non-ASCII di Windows tidak menjadi masalah dan tidak perlu berkas sementara.

    Mode orientasi EXIF sengaja TIDAK diterapkan, sama seperti layanan
    sebelumnya. Bingkai dari halaman kamera selalu keluar dari kanvas peramban
    yang tidak membawa penanda orientasi, jadi jalur ini tidak terpengaruh.
    """
    import cv2
    import numpy as np

    try:
        buffer = np.frombuffer(data, dtype=np.uint8)
        gambar = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as galat:  # noqa: BLE001
        logger.warning("gambar tidak bisa dibaca: %s", galat)

        return None

    if gambar is None or getattr(gambar, "size", 0) == 0:
        return None

    tinggi, lebar = gambar.shape[:2]

    if tinggi <= 0 or lebar <= 0:
        return None

    if MAX_IMAGE_PIXELS > 0 and tinggi * lebar > MAX_IMAGE_PIXELS:
        logger.warning("gambar terlalu besar: %sx%s piksel", lebar, tinggi)

        return None

    return gambar


def detect_size_for_frame(width: int, height: int, long_side: int, multiple: int = 32) -> tuple[int, int]:
    """Hitung bentuk kanvas detektor yang rasinya SAMA dengan gambar.

    Detektor mengisi kanvas berukuran input_size dengan padding nol bila rasio
    gambar berbeda. Untuk kamera 16:9 pada kanvas 640x640, hanya 640x360 yang
    berisi gambar sehingga 43,75 persen kanvas adalah nol yang TETAP dihitung
    model. Mengikuti rasio gambar menghilangkan pemborosan itu.

    Sisi panjang dipertahankan supaya detail wajah tidak hilang, sisi pendek
    dibulatkan ke kelipatan `multiple` agar petak fitur (stride 8 sampai 128)
    tetap rapi.
    """
    if width <= 0 or height <= 0 or long_side <= 0:
        return (640, 640)

    if width >= height:
        out_w = long_side
        out_h = int(round(long_side * height / width))
    else:
        out_h = long_side
        out_w = int(round(long_side * width / height))

    out_w = max(multiple, int(round(out_w / multiple)) * multiple)
    out_h = max(multiple, int(round(out_h / multiple)) * multiple)

    return (out_w, out_h)


def proof_photo(frame: "Any", max_side: int = PROOF_MAX_SIDE) -> dict[str, Any] | None:
    """Perkecil seluruh bingkai menjadi JPEG kecil untuk bukti absen.

    Sisi terpanjangnya diperkecil lebih dulu, gambar kecil tidak pernah
    diperbesar, lalu mutu JPEG diturunkan bertahap sampai masuk batas ukuran.
    Mengembalikan None bila pengolahannya gagal, karena bukti absen tidak boleh
    menggagalkan absennya.
    """
    import cv2
    from PIL import Image

    try:
        gambar = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        foto = Image.fromarray(gambar)
        foto.thumbnail((max_side, max_side), Image.LANCZOS)

        for mutu in PROOF_QUALITIES:
            buffer = io.BytesIO()
            foto.save(buffer, format="JPEG", quality=mutu, optimize=True, progressive=True)
            ukuran = buffer.tell()

            if PROOF_MAX_BYTES <= 0 or ukuran <= PROOF_MAX_BYTES:
                return {
                    "photo": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "photo_bytes": ukuran,
                    "photo_width": foto.width,
                    "photo_height": foto.height,
                    "photo_quality": mutu,
                }

        # Semua mutu masih kelebihan: pakai yang terkecil supaya absen tetap
        # punya bukti, dan catat kejadiannya untuk operator.
        logger.warning("foto bukti masih %s bita pada mutu terendah", ukuran)

        return {
            "photo": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "photo_bytes": ukuran,
            "photo_width": foto.width,
            "photo_height": foto.height,
            "photo_quality": PROOF_QUALITIES[-1],
        }
    except Exception as galat:  # noqa: BLE001
        logger.warning("foto bukti gagal dibuat: %s", galat)

        return None


# =====================================================================
# MESIN WAJAH
# =====================================================================


class FaceEngine:
    """Pemegang model InsightFace. Satu salinan dipakai seluruh permintaan."""

    def __init__(self) -> None:
        self.analyzer: Any | None = None
        self.error: str | None = None
        self.loaded_at: float | None = None
        self.load_seconds: float | None = None
        self.calls = 0

    @property
    def ready(self) -> bool:
        return self.analyzer is not None

    def load(self) -> None:
        """Muat model. Unduhan terjadi di sini pada pemakaian pertama."""
        if self.ready:
            return

        mulai = time.perf_counter()

        try:
            apply_runtime_limits()

            from insightface.app import FaceAnalysis

            if USE_GPU:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = GPU_DEVICE_ID
            else:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1

            MODEL_ROOT.mkdir(parents=True, exist_ok=True)
            logger.info("memuat model %s dari %s (providers=%s)", MODEL_NAME, MODEL_ROOT, providers)

            analyzer = FaceAnalysis(
                name=MODEL_NAME,
                root=str(MODEL_ROOT),
                providers=providers,
                allowed_modules=["detection", "recognition"],
            )
            analyzer.prepare(ctx_id=ctx_id, det_size=(DETECT_LONG_SIDE, DETECT_LONG_SIDE), det_thresh=DET_THRESHOLD)

            self.analyzer = analyzer
            self.error = None
            self.loaded_at = time.time()
            self.load_seconds = time.perf_counter() - mulai

            logger.info("model siap dalam %.2f detik", self.load_seconds)

            if WARMUP:
                self.warmup()
        except Exception as galat:  # noqa: BLE001
            self.analyzer = None
            self.error = f"{type(galat).__name__}: {galat}"
            logger.error("model gagal dimuat: %s", self.error)

    def warmup(self) -> None:
        """Jalankan deteksi sekali pada gambar kosong.

        Tujuannya membayar biaya penyiapan pustaka sekali saja. Pengenalan tidak
        bisa dipanaskan tanpa gambar berisi wajah, jadi hanya detektor yang
        dipanaskan di sini dan itu ditulis apa adanya.
        """
        import numpy as np

        if self.analyzer is None:
            return

        try:
            mulai = time.perf_counter()
            kosong = np.zeros((DETECT_LONG_SIDE, DETECT_LONG_SIDE, 3), dtype=np.uint8)
            self.analyzer.det_model.detect(kosong, max_num=0, metric="default")
            logger.info("pemanasan detektor selesai dalam %.1f ms", (time.perf_counter() - mulai) * 1000)
        except Exception as galat:  # noqa: BLE001
            logger.warning("pemanasan gagal, layanan tetap jalan: %s", galat)

    def analyze(
        self,
        image: "Any",
        *,
        need_embeddings: bool = True,
        min_face_size: int = EMBED_MIN_FACE_SIZE,
    ) -> list[dict[str, Any]]:
        """Deteksi semua wajah, lalu hitung angkanya HANYA bila akan dipakai.

        Kenapa tidak memakai `analyzer.get()` bawaan insightface: fungsi itu
        selalu menjalankan model pengenalan untuk setiap wajah, padahal jawabannya
        sering langsung dibuang - saat wajah terlalu kecil untuk dinilai, atau
        saat bingkai memuat lebih banyak wajah daripada batas. Di sini keputusan
        itu diambil SEBELUM model pengenalan dijalankan.

        Urutan hasil mengikuti besar kotak, dari yang terbesar, karena wajah
        terbesar biasanya orang yang paling dekat dengan kamera.
        """
        import numpy as np
        from insightface.app.common import Face

        if self.analyzer is None:
            return []

        self.calls += 1

        tinggi, lebar = image.shape[:2]
        ukuran = (
            detect_size_for_frame(lebar, tinggi, DETECT_LONG_SIDE)
            if DETECT_MATCH_ASPECT
            else (DETECT_LONG_SIDE, DETECT_LONG_SIDE)
        )

        bboxes, kpss = self.analyzer.det_model.detect(
            image, max_num=0, metric="default", input_size=ukuran
        )

        if bboxes is None or bboxes.shape[0] == 0:
            return []

        # Wajah terbesar lebih dulu. Kotak dari detektor sudah dalam koordinat
        # gambar asli, jadi tidak perlu dikembalikan ke skala mana pun.
        urutan = sorted(
            range(bboxes.shape[0]),
            key=lambda i: float((bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])),
            reverse=True,
        )

        if MAX_FACES > 0:
            urutan = urutan[:MAX_FACES]

        hasil: list[dict[str, Any]] = []

        for index in urutan:
            x1, y1, x2, y2 = (int(round(float(nilai))) for nilai in bboxes[index, 0:4])
            det_score = float(bboxes[index, 4])
            sisi = max(0, min(x2 - x1, y2 - y1))
            terlalu_kecil = min_face_size > 0 and sisi < min_face_size

            encoding: list[float] | None = None
            kesalahan: str | None = None

            if terlalu_kecil:
                # Tidak dihitung angkanya: satu wajah kecil bisa memakan waktu
                # sebesar wajah besar, sedangkan angkanya tidak layak dipercaya.
                kesalahan = "face_too_small"
            elif need_embeddings:
                try:
                    face = Face(
                        bbox=bboxes[index, 0:4],
                        kps=None if kpss is None else kpss[index],
                        det_score=det_score,
                    )

                    for taskname, model in self.analyzer.models.items():
                        if taskname == "detection":
                            continue
                        model.get(image, face)  # mengisi face.embedding

                    mentah = getattr(face, "embedding", None)

                    if mentah is not None and len(mentah) > 0:
                        # Angka InsightFace panjangnya sekitar 27, jadi WAJIB
                        # dinormalkan. Dengan panjang 1, jarak euclid antar dua
                        # angka setara dengan cosine similarity, sehingga aplikasi
                        # bisa membandingkannya sebagai jarak.
                        vektor = np.asarray(mentah, dtype=np.float32).ravel()
                        norma = float(np.linalg.norm(vektor))

                        if norma > 0 and np.isfinite(norma):
                            encoding = [round(float(nilai), 6) for nilai in (vektor / norma)]
                        else:
                            kesalahan = "encoding_failed"
                    else:
                        kesalahan = "encoding_failed"
                except Exception as galat:  # noqa: BLE001
                    logger.warning("angka wajah gagal dihitung: %s", galat)
                    kesalahan = "encoding_failed"

            hasil.append(
                {
                    "top": y1,
                    "right": x2,
                    "bottom": y2,
                    "left": x1,
                    "face_size": sisi,
                    "det_score": round(det_score, 4),
                    "encoding": encoding,
                    "error": kesalahan,
                }
            )

        return hasil


def apply_runtime_limits(
    intra_threads: int = ORT_INTRA_THREADS,
    inter_threads: int = ORT_INTER_THREADS,
    opencv_threads: int = OPENCV_THREADS,
) -> None:
    """Batasi jumlah thread agar OpenCV dan onnxruntime tidak berebut core.

    Kenapa perlu menambal: `insightface.model_zoo.model_zoo.get_model()` hanya
    meneruskan `providers` dan `provider_options` ke `PickableInferenceSession`,
    sehingga argumen `sess_options=` yang dikirim ke `FaceAnalysis(...)` dibuang
    TANPA peringatan apa pun. Jadi jumlah thread tidak bisa diatur lewat jalur
    resmi itu dan kelas sesinya perlu ditambal tipis di sini.

    Harus dipanggil SEBELUM FaceAnalysis dibuat, karena sesinya lahir di sana.
    """
    import cv2

    cv2.setUseOptimized(True)

    if opencv_threads > 0:
        cv2.setNumThreads(opencv_threads)

    if intra_threads <= 0 and inter_threads <= 0:
        return

    import onnxruntime
    import insightface.model_zoo.model_zoo as model_zoo

    kelas = model_zoo.PickableInferenceSession
    asli = getattr(kelas, "_layanan_wajah_original_init", None)

    if asli is None:
        asli = kelas.__init__
        kelas._layanan_wajah_original_init = asli

    opsi = onnxruntime.SessionOptions()

    if intra_threads > 0:
        opsi.intra_op_num_threads = intra_threads

    if inter_threads > 0:
        opsi.inter_op_num_threads = inter_threads

    opsi.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    def init_bertambal(self, model_path, **kwargs):
        kwargs.setdefault("sess_options", opsi)
        asli(self, model_path, **kwargs)

    kelas.__init__ = init_bertambal
    logger.info(
        "batas thread: onnxruntime intra=%s inter=%s, opencv=%s",
        intra_threads,
        inter_threads,
        opencv_threads,
    )


ENGINE = FaceEngine()


# =====================================================================
# LAYANAN HTTP
# =====================================================================


def _balasan_refusal(status: str, message: str, face_count: int = 0, face_size: int = 0) -> dict[str, Any]:
    """Balasan untuk foto yang tidak bisa dipakai."""
    return {
        "status": status,
        "encoding": None,
        "face_count": face_count,
        "face_size": face_size,
        "model": MODEL_LABEL,
        "message": message,
    }


def _iris_model() -> str:
    """Nama model untuk /detect, yang tidak memakai kunci encoding tunggal."""
    return MODEL_LABEL


def _terapkan_opsi(balasan: dict[str, Any], frame: "Any", with_photo: bool) -> dict[str, Any]:
    """Tambahkan foto bukti ke balasan bila diminta dan berhasil dibuat."""
    if with_photo:
        bukti = proof_photo(frame)

        if bukti is not None:
            balasan.update(bukti)

    return balasan


def build_app() -> "Any":
    """Rakit aplikasi FastAPI."""

    @asynccontextmanager
    async def lifespan(app: "Any") -> Iterator[None]:
        ENGINE.load()
        yield

    app = FastAPI(
        title="Layanan Wajah Absen Solat (InsightFace)",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> JSONResponse:
        """Penanda layanan siap.

        Balasannya 503 bila modelnya belum siap, dan itu disengaja: aplikasi
        sekolah memakai kode HTTP untuk menentukan apakah fitur wajah bisa
        dipakai, jadi model yang belum termuat tidak boleh terlihat siap.
        """
        siap = ENGINE.ready
        badan = {
            "status": "ok" if siap else ("memuat" if ENGINE.error is None else "gagal"),
            "model": MODEL_LABEL,
            "dimensions": str(DIMENSIONS),
            "model_ready": siap,
            "model_error": ENGINE.error,
            "load_seconds": None if ENGINE.load_seconds is None else round(ENGINE.load_seconds, 2),
            "calls": ENGINE.calls,
            "detector": {
                "long_side": DETECT_LONG_SIDE,
                "match_aspect": DETECT_MATCH_ASPECT,
                "threshold": DET_THRESHOLD,
                "max_faces": MAX_FACES,
            },
            "threads": {
                "onnxruntime_intra": ORT_INTRA_THREADS,
                "onnxruntime_inter": ORT_INTER_THREADS,
                "opencv": OPENCV_THREADS,
            },
        }

        return JSONResponse(badan, status_code=200 if siap else 503)

    @app.post("/encode")
    async def encode(
        image: UploadFile = File(..., description="Berkas foto, field bernama image"),
        with_photo: str | None = Query(None),
        require_single: str | None = Query(None),
    ) -> JSONResponse:
        """Hitung angka wajah untuk SATU orang.

        Dipakai pendaftaran wajah dan perangkat sensor gerbang. Bila foto memuat
        lebih dari satu wajah, yang dipakai adalah wajah terbesar, kecuali
        `require_single=1` diminta: saat itu foto ditolak supaya alur absen tidak
        diam-diam memakai wajah orang lain.
        """
        minta_bukti = _benar(with_photo)
        satu_saja = _benar(require_single)

        if not ENGINE.ready:
            return JSONResponse(
                {
                    "status": "error",
                    "encoding": None,
                    "face_count": 0,
                    "face_size": 0,
                    "model": MODEL_LABEL,
                    "message": "Model wajah belum siap di layanan. Coba lagi sebentar.",
                },
                status_code=503,
            )

        data = await image.read()
        frame = decode_image(data)

        if frame is None:
            return JSONResponse(_balasan_refusal("no_face", "Foto tidak terbaca. Ambil ulang fotonya."), status_code=400)

        mulai = time.perf_counter()
        # Pada jalur satu orang, angka wajah selalu dihitung: hanya ada satu
        # wajah, jadi menghemat di sini tidak ada gunanya.
        wajah = ENGINE.analyze(frame, need_embeddings=True, min_face_size=0)
        detik = time.perf_counter() - mulai

        if not wajah:
            return JSONResponse(_balasan_refusal("no_face", "Wajah tidak terlihat."))

        if satu_saja and len(wajah) > 1:
            return JSONResponse(
                _balasan_refusal(
                    "multiple_faces",
                    f"Ada {len(wajah)} wajah di foto. Absen satu orang saja supaya tidak salah catat.",
                    face_count=len(wajah),
                    face_size=wajah[0]["face_size"],
                )
            )

        utama = wajah[0]

        if utama["encoding"] is None:
            return JSONResponse(
                _balasan_refusal(
                    "face_too_small",
                    "Wajah terlalu kecil atau tidak bisa dihitung. Dekatkan wajah ke kamera.",
                    face_count=len(wajah),
                    face_size=utama["face_size"],
                )
            )

        logger.info(
            "encode %sx%s: %s wajah dalam %.1f ms",
            frame.shape[1],
            frame.shape[0],
            len(wajah),
            detik * 1000,
        )

        balasan = {
            "status": "success",
            "encoding": utama["encoding"],
            "face_count": len(wajah),
            "face_size": utama["face_size"],
            "model": _iris_model(),
            "message": "Wajah terbaca.",
        }

        return JSONResponse(_terapkan_opsi(balasan, frame, minta_bukti))

    @app.post("/detect")
    async def detect(
        image: UploadFile = File(..., description="Berkas foto, field bernama image"),
        with_photo: str | None = Query(None),
    ) -> JSONResponse:
        """Hitung angka wajah untuk SELURUH wajah di satu bingkai.

        Dipakai kamera absen yang bisa memuat beberapa orang sekaligus. Setiap
        wajah dilaporkan sendiri sendiri beserta kotaknya. Kotak memakai
        koordinat piksel gambar ASLI yang dikirim pemanggil, sehingga bisa
        digambar apa adanya di atas bingkai kamera.

        Wajah yang angkanya gagal dihitung tetap dilaporkan dengan `encoding`
        bernilai null, supaya satu wajah yang gagal tidak menggagalkan wajah
        lain dan halaman tetap bisa menggambar kotaknya.
        """
        minta_bukti = _benar(with_photo)

        kosong: dict[str, Any] = {
            "status": "no_face",
            "face_count": 0,
            "width": 0,
            "height": 0,
            "model": MODEL_LABEL,
            "message": "Wajah tidak terlihat.",
            "faces": [],
        }

        if not ENGINE.ready:
            return JSONResponse({**kosong, "status": "error", "message": "Model wajah belum siap di layanan."}, status_code=503)

        data = await image.read()
        frame = decode_image(data)

        if frame is None:
            return JSONResponse({**kosong, "status": "error", "message": "Foto tidak terbaca. Ambil ulang fotonya."}, status_code=400)

        tinggi, lebar = frame.shape[:2]

        mulai = time.perf_counter()
        wajah = ENGINE.analyze(frame, need_embeddings=True, min_face_size=EMBED_MIN_FACE_SIZE)
        detik = time.perf_counter() - mulai

        if not wajah:
            return JSONResponse({**kosong, "width": lebar, "height": tinggi})

        daftar = []

        for index, satu in enumerate(wajah):
            daftar.append(
                {
                    "index": index,
                    "bbox": {
                        "top": satu["top"],
                        "right": satu["right"],
                        "bottom": satu["bottom"],
                        "left": satu["left"],
                    },
                    "face_size": satu["face_size"],
                    "encoding": satu["encoding"],
                    "error": satu["error"],
                }
            )

        logger.info(
            "detect %sx%s: %s wajah dalam %.1f ms",
            lebar,
            tinggi,
            len(daftar),
            detik * 1000,
        )

        balasan = {
            "status": "success",
            "face_count": len(daftar),
            "width": lebar,
            "height": tinggi,
            "model": MODEL_LABEL,
            "message": "Wajah terbaca." if len(daftar) == 1 else f"{len(daftar)} wajah terbaca.",
            "faces": daftar,
        }

        return JSONResponse(_terapkan_opsi(balasan, frame, minta_bukti))

    return app


# =====================================================================
# BARIS PERINTAH
# =====================================================================


def print_check() -> int:
    """Cetak versi pustaka dan pengaturan yang sedang berlaku."""
    print("Pengaturan layanan")
    print(f"  alamat                : {HOST}:{PORT}")
    print(f"  model                 : {MODEL_NAME} (dilaporkan sebagai {MODEL_LABEL})")
    print(f"  folder model          : {MODEL_ROOT}")
    print(f"  dimensi angka wajah   : {DIMENSIONS}")
    print(f"  sisi panjang detektor : {DETECT_LONG_SIDE} (ikut rasio: {DETECT_MATCH_ASPECT})")
    print(f"  ambang detektor       : {DET_THRESHOLD}")
    print(f"  batas wajah per foto  : {MAX_FACES}")
    print(f"  batas wajah terkecil  : {EMBED_MIN_FACE_SIZE} px")
    print(f"  thread                : intra {ORT_INTRA_THREADS}, inter {ORT_INTER_THREADS}, opencv {OPENCV_THREADS}")
    print(f"  foto bukti            : sisi {PROOF_MAX_SIDE}, mutu {PROOF_QUALITY}, batas {PROOF_MAX_KB} KB")
    print()
    print("Pustaka")

    for nama in ("cv2", "numpy", "onnxruntime", "insightface", "PIL", "fastapi", "uvicorn"):
        try:
            modul = __import__(nama)
            versi = getattr(modul, "__version__", "tidak dilaporkan")
            print(f"  {nama:14s}: {versi}")
        except Exception as galat:  # noqa: BLE001
            print(f"  {nama:14s}: TIDAK ADA ({type(galat).__name__})")

    return 0


def print_preload() -> int:
    """Unduh dan muat model sekali, lalu keluar.

    Dipakai supaya unduhan besar terjadi saat menyiapkan server atau saat
    membangun citra, bukan saat guru pertama mencoba absen.
    """
    ENGINE.load()

    if not ENGINE.ready:
        print(f"model GAGAL dimuat: {ENGINE.error}")

        return 1

    print(f"model {MODEL_LABEL} siap, dimuat dalam {ENGINE.load_seconds:.2f} detik")
    print(f"folder model: {MODEL_ROOT}")

    return 0


def main(argv: list[str] | None = None) -> int:
    argumen = list(sys.argv[1:] if argv is None else argv)

    if "--check" in argumen or "--versi" in argumen:
        return print_check()

    if "--preload" in argumen or "--unduh-model" in argumen:
        return print_preload()

    import uvicorn

    uvicorn.run(
        build_app(),
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL.lower(),
        access_log=False,
        workers=1,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
