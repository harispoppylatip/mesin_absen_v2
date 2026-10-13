# =====================================================================
# Layanan wajah InsightFace untuk aplikasi absen solat
#
# Cara memakai:
#     docker compose build
#     docker compose up -d
#     docker compose logs -f
#
# Dua tahap, dan pemisahan itu disengaja:
#
#   pembangun   memasang pustaka, menambal insightface, dan MENGUNDUH MODEL.
#               Model ikut masuk citra supaya layanan bisa jalan tanpa internet
#               dan tidak ada kejutan mengunduh saat guru pertama mencoba absen.
#               Model buffalo_sc hanya sekitar 16 MB, jadi tidak mahal.
#   layanan     hanya memuat hasil pemasangan dan berkas yang diperlukan untuk
#               jalan. Tidak ada compiler, tidak ada cache pip, tidak ada
#               berkas sumber. Isinya jauh lebih kecil daripada tahap pertama.
#
# KENAPA TIDAK ADA build-essential
# ---------------------------------
# insightface 0.7.3 hanya dikirim sebagai kode sumber dan selalu mencoba
# mengompilasi satu ekstensi C++ (`mesh_core_cython`) yang HANYA dipakai untuk
# render topeng wajah 3D. Deteksi dan pengenalan wajah tidak memakainya.
# Skrip `tambal_insightface.py` mematikan ekstensi itu dan memasang pengganti
# Python murni, sehingga compiler tidak diperlukan sama sekali. Menghapus
# build-essential menghemat sekitar 300 MB dari citra akhir.
# =====================================================================

# =====================================================================
# TAHAP 1: PEMBANGUN
# =====================================================================
FROM python:3.10-slim-bookworm AS pembangun

# Python 3.10 dipilih karena itu versi yang dipakai saat seluruh pengukuran di
# berkas catatan dilakukan, sehingga versi numpy, onnxruntime, dan opencv yang
# terkunci di requirements.txt sudah terbukti cocok.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /bangun

# requirements.txt disalin lebih dulu dan dipasang terpisah dari kode. Dengan
# begitu, mengubah app.py tidak membatalkan pemasangan pustaka yang memakan
# waktu paling lama.
COPY requirements.txt ./

RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Tambalan insightface. Satu berkas ini dipakai bersama Windows dan Linux, jadi
# tidak ada dua salinan logika tambalan yang harus diubah bersamaan.
COPY tambal_insightface.py ./

RUN python tambal_insightface.py

# Model diunduh pada waktu bangun. `--preload` memuat model lalu keluar, jadi
# galat unduhan atau galat model terlihat saat membangun citra, bukan saat
# layanan dipakai.
COPY app.py ./

ENV FACE_MODEL=buffalo_sc \
    FACE_MODEL_ROOT=/model \
    FACE_DIMENSIONS=512 \
    FACE_WARMUP=0

RUN python app.py --preload \
 && du -sh /model

# =====================================================================
# TAHAP 2: LAYANAN
# =====================================================================
FROM python:3.10-slim-bookworm AS layanan

# libgomp1 WAJIB: onnxruntime memakai OpenMP untuk menjalankan beberapa thread,
# dan pustaka itu tidak ikut dibundel di dalam roda manylinux-nya. Tanpa
# libgomp1, `import onnxruntime` gagal dengan
#     ImportError: libgomp.so.1: cannot open shared object file
#
# libglib2.0-0 disediakan sebagai jaring pengaman opencv. Versi headless
# umumnya tidak memerlukannya, tetapi memasang paket kecil ini jauh lebih murah
# daripada citra yang gagal start hanya karena satu pustaka bersama.
RUN apt-get update \
 && apt-get install --no-install-recommends --yes \
      libgomp1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FACE_HOST=0.0.0.0 \
    FACE_PORT=5005 \
    FACE_MODEL=buffalo_sc \
    FACE_MODEL_NAME=insightface-buffalo_sc \
    FACE_DIMENSIONS=512 \
    FACE_MODEL_ROOT=/model \
    FACE_DETECT_LONG_SIDE=640 \
    FACE_DETECT_MATCH_ASPECT=1 \
    FACE_MAX_FACES=8 \
    FACE_ORT_INTRA_THREADS=4 \
    FACE_ORT_INTER_THREADS=1 \
    FACE_OPENCV_THREADS=1

# Layanan dijalankan sebagai pengguna biasa, bukan root. Uid 10001 dipilih
# supaya tidak bertabrakan dengan pengguna bawaan citra.
RUN groupadd --system --gid 10001 wajah \
 && useradd --system --uid 10001 --gid wajah --home-dir /srv --shell /usr/sbin/nologin wajah

COPY --from=pembangun /opt/venv /opt/venv
COPY --from=pembangun /model /model

WORKDIR /srv

COPY app.py verifikasi.py ./

RUN chown -R wajah:wajah /srv /model

USER wajah

EXPOSE 5005

# Layanan ini tidak punya kata sandi. Membukanya ke jaringan harus keputusan
# sadar operator, jadi port hanya dipublikasikan ke 127.0.0.1 secara bawaan
# (diatur di docker-compose.yml, bukan di sini).
#
# start-period sengaja longgar: memuat model butuh 4 detik di mesin
# pengembang, dan bisa jauh lebih lama pada CPU server yang lemah atau saat
# berkas model belum ada di singgahan disk.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import os,sys,urllib.request; u='http://127.0.0.1:'+os.getenv('FACE_PORT','5005')+'/health'; sys.exit(0 if urllib.request.urlopen(u,timeout=8).status==200 else 1)"

CMD ["python", "app.py"]
