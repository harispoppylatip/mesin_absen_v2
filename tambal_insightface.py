"""Pasang insightface 0.7.3 tanpa compiler C++.

KENAPA BERKAS INI ADA
---------------------
`pip install insightface==0.7.3` selalu mengunduh berkas SUMBER, bukan roda
siap pakai, lalu mencoba mengompilasi satu ekstensi C++ bernama
`mesh_core_cython`. Ekstensi itu HANYA dipakai untuk render topeng wajah 3D
(`insightface.app.mask_renderer`). Deteksi wajah (`scrfd`) dan pengenalan wajah
(`arcface`) tidak memakainya sama sekali.

Akibatnya pemasangan gagal di dua tempat sekaligus:

  * Windows tanpa Windows 10 SDK
        fatal error C1083: Cannot open include file: 'stdio.h'
  * `python:3.10-slim` di dalam Docker, yang memang tidak membawa compiler
        error: command 'gcc' failed

Skrip ini menempuh jalan yang sama di kedua tempat: mematikan ekstensi C++ itu,
lalu memasang modul Python murni sebagai penggantinya supaya rantai import
`insightface` tetap utuh.

KENAPA DIPISAH DARI Dockerfile
------------------------------
Logika tambalan ini dulu ada di `install-insightface.ps1` untuk Windows. Kalau
hal yang sama ditulis ulang di dalam Dockerfile, akan ada DUA salinan logika
yang wajib diubah bersamaan. Berkas ini menjadi satu-satunya salinan itu, dan
dipakai oleh Windows maupun Linux.

PEMAKAIAN
---------
    pip install -r requirements.txt        lalu     python tambal_insightface.py

URUTANNYA WAJIB, dan alasannya begini: requirements.txt SENGAJA tidak memuat
insightface. Kalau paket itu ditulis di sana, `pip install -r requirements.txt`
akan mencoba mengompilasi ekstensi C++-nya sendiri dan gagal sebelum skrip ini
sempat berjalan, persis seperti yang pernah terjadi di server:

    error: [Errno 2] No such file or directory: 'g++'
    ERROR: Failed building wheel for insightface

Pemakaian langsung:
    python tambal_insightface.py                  pasang memakai python ini
    python tambal_insightface.py --pip <python>   pasang ke python lain
    python tambal_insightface.py --kerja <folder> tentukan folder kerja
    python tambal_insightface.py --sisakan        jangan hapus folder kerja

Alternatifnya di Windows, tanpa compiler, sudah tersedia `install-insightface.ps1`
di folder induk. Keduanya menghasilkan pemasangan yang sama.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

VERSI = "0.7.3"
POLA_EKSTENSI = "ext_modules=cythonize(extensions)"
POLA_MATI = "ext_modules=[]"

ISI_INIT = '''"""Pembungkus direktori `mesh/cython`.

Direktori ini semula BUKAN paket Python (tidak ada `__init__.py`) karena
isinya hanya berkas .pyx/.cpp yang dikompilasi menjadi satu modul ekstensi
`mesh_core_cython`.

Berkas ini ditambahkan oleh tambalan `tambal_insightface.py` supaya direktori
ikut dikenali `find_packages()` dan `mesh_core_cython.py` (pengganti Python
murni) benar-benar terpasang di dalam paket.
"""
'''

ISI_STUB = '''"""Pengganti Python murni untuk ekstensi C++ `mesh_core_cython`.

MENGAPA BERKAS INI ADA
----------------------
`insightface` 0.7.3 mengirim satu ekstensi C++ bernama `mesh_core_cython`
(sumber `mesh_core_cython.pyx` + `mesh_core.cpp`). Ekstensi itu hanya dipakai
oleh `insightface.thirdparty.face3d.mesh` untuk urusan topeng wajah 3D, yaitu
`io.py`, `light.py`, dan `render.py` - bukan oleh deteksi wajah (`scrfd` /
`retinaface`) maupun pengenalan wajah (`arcface`).

Masalahnya, berkas berkas itu menulis `from .cython import mesh_core_cython`
TANPA `try/except`, sehingga seluruh paket `insightface` gagal di-import kalau
modul ekstensi tersebut tidak ada. Karena deteksi dan pengenalan wajah tidak
memerlukannya, modul ini dipasang sebagai pengganti agar rantai import
`insightface` tetap utuh tanpa perlu compiler C++.

AKIBATNYA
---------
Deteksi dan pengenalan wajah berjalan normal. Yang tidak tersedia hanya render
topeng wajah 3D (`insightface.app.mask_renderer`). Setiap fungsi di sini akan
melempar `NotImplementedError` bila benar-benar dipanggil.
"""

__all__: list[str] = []

_PESAN = (
    "mesh_core_cython (ekstensi C++ insightface) tidak dibangun di komputer ini "
    "karena compiler C++ tidak tersedia. Modul ini hanya pengganti kosong: "
    "deteksi dan pengenalan wajah tetap berjalan, tetapi render topeng wajah 3D "
    "(insightface.app.mask_renderer / face3d.mesh) tidak tersedia."
)


def __getattr__(name: str):
    """Gagalkan dengan pesan yang jelas bila ada kode yang memanggil fungsinya."""
    raise NotImplementedError(f"{_PESAN} Fungsi yang diminta: {name}()")
'''

# Program kecil yang dijalankan sesudah pemasangan berhasil. Isinya SENGAJA
# menyentuh tiga hal: paket utamanya, pemakaian nyatanya (FaceAnalysis), dan
# modul penggantinya. Kalau salah satu hilang, verifikasi ini gagal.
VERIFIKASI = """import insightface
from insightface.app import FaceAnalysis
from insightface.thirdparty.face3d.mesh.cython import mesh_core_cython
print('    insightface', insightface.__version__, 'siap dipakai')
print('    FaceAnalysis tersedia:', bool(FaceAnalysis))
print('    pengganti mesh_core_cython terpasang di', mesh_core_cython.__file__)
"""


def jalan(perintah: list[str], keterangan: str) -> None:
    """Jalankan satu perintah, hentikan bila gagal."""
    print(f"  {keterangan}")

    hasil = subprocess.run(perintah, capture_output=True, text=True)

    if hasil.returncode != 0:
        print(hasil.stdout[-4000:])
        print(hasil.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"gagal: {' '.join(perintah)}")


def verifikasi(pip: str) -> None:
    """Pastikan paketnya benar benar bisa diimpor SESUDAH dipasang.

    Sengaja mengimpor rantai penuhnya, termasuk modul pengganti, sebab dua
    kegagalan yang paling mudah terjadi justru TIDAK terlihat dari sekadar
    `import insightface`:

      * berkas pengganti tidak ikut terpasang ke dalam paket, sehingga
        `from .cython import mesh_core_cython` gagal. Ini bisa terjadi bila
        direktori itu tidak dikenali sebagai paket saat roda dibangun.
      * ada pustaka yang dibutuhkan SAAT IMPOR tetapi belum terpasang. Yang
        paling mudah terlewat adalah matplotlib, sebab berkas yang memakainya
        (`thirdparty/face3d/mesh/vis.py`) tersembunyi jauh di dalam rantai:
        insightface.app -> mask_renderer -> thirdparty.face3d -> mesh -> vis.
    """
    print("  memverifikasi ...")

    hasil = subprocess.run([pip, "-c", VERIFIKASI], capture_output=True, text=True)

    if hasil.returncode != 0:
        print(hasil.stdout[-2000:])
        print(hasil.stderr[-4000:], file=sys.stderr)
        print()
        print("  insightface terpasang, tetapi paketnya belum bisa diimpor.")
        print("  Periksa dua hal ini lebih dulu:")
        print("    1. seluruh isi requirements.txt sudah terpasang, terutama matplotlib")
        print("    2. berkas pengganti mesh_core_cython ikut terpasang ke dalam paket")
        raise SystemExit("verifikasi gagal")

    print(hasil.stdout.rstrip())


def tambal(sumber: Path) -> Path:
    """Matikan ekstensi C++ dan pasang pengganti Python murni."""
    setup = sumber / "setup.py"
    teks = setup.read_text(encoding="utf-8")

    if POLA_MATI in teks:
        print("  setup.py sudah bertambalan")
    elif POLA_EKSTENSI in teks:
        teks = teks.replace(
            POLA_EKSTENSI,
            "# TAMBALAN tambal_insightface.py: ekstensi C++ dimatikan\n" + POLA_MATI,
        )
        setup.write_text(teks, encoding="utf-8")
        print("  setup.py: ekstensi C++ dimatikan")
    else:
        raise SystemExit(
            f"pola '{POLA_EKSTENSI}' tidak ditemukan di {setup}. "
            "Versi insightface mungkin berbeda, periksa manual."
        )

    cython = sumber / "insightface" / "thirdparty" / "face3d" / "mesh" / "cython"

    if not cython.is_dir():
        raise SystemExit(f"folder mesh/cython tidak ditemukan: {cython}")

    # Ditulis dengan akhir baris LF dan tanpa penanda urutan bita, supaya berkas
    # yang sama dipakai apa adanya di Windows maupun di dalam Docker.
    (cython / "__init__.py").write_text(ISI_INIT, encoding="utf-8", newline="\n")
    (cython / "mesh_core_cython.py").write_text(ISI_STUB, encoding="utf-8", newline="\n")
    print("  pengganti mesh_core_cython.py dipasang")

    return sumber


def main() -> int:
    parser = argparse.ArgumentParser(description="Pasang insightface tanpa compiler C++.")
    parser.add_argument("--pip", default=sys.executable, help="python yang dipakai memasang")
    parser.add_argument("--kerja", default=None, help="folder kerja sementara")
    parser.add_argument("--sumber", default=None, help="folder sumber yang sudah diekstrak")
    parser.add_argument("--sisakan", action="store_true", help="jangan hapus folder kerja")
    argumen = parser.parse_args()

    print(f"=== Pasang insightface {VERSI} tanpa compiler C++ ===")

    milik_sendiri = argumen.kerja is None and argumen.sumber is None
    kerja = Path(argumen.kerja) if argumen.kerja else Path(tempfile.mkdtemp(prefix="insightface-"))

    if kerja.exists() and argumen.kerja:
        shutil.rmtree(kerja, ignore_errors=True)

    kerja.mkdir(parents=True, exist_ok=True)

    try:
        if argumen.sumber:
            sumber = Path(argumen.sumber)
        else:
            print("  mengunduh berkas sumber ...")
            # --no-build-isolation juga WAJIB di langkah unduh ini, bukan hanya
            # saat memasang.
            #
            # Tanpanya pip membuat lingkungan terpisah dan MENGUNDUH ulang
            # setuptools, numpy, serta cython hanya untuk bisa membaca metadata
            # paketnya. Unduhan tambahan itu membuat langkah ini rapuh tanpa
            # manfaat apa pun, dan sudah pernah gagal di percobaan pertama:
            #
            #     Installing build dependencies: started
            #     Installing build dependencies: finished with status 'error'
            #
            # lalu berhasil begitu dijalankan ulang. Seluruh isi [build-system]
            # insightface sudah dipasang oleh requirements.txt tepat sebelum
            # langkah ini, jadi lingkungan terpisah memang tidak diperlukan.
            jalan(
                [
                    argumen.pip,
                    "-m",
                    "pip",
                    "download",
                    f"insightface=={VERSI}",
                    "--no-deps",
                    "--no-build-isolation",
                    "-d",
                    str(kerja),
                ],
                "unduh selesai",
            )

            berkas = next(iter(sorted(kerja.glob("insightface-*.tar.gz"))), None)

            if berkas is None:
                raise SystemExit(f"berkas sumber tidak ditemukan di {kerja}")

            with tarfile.open(berkas) as arsip:
                arsip.extractall(kerja)

            sumber = next(p for p in sorted(kerja.iterdir()) if p.is_dir() and p.name.startswith("insightface-"))

        print(f"  sumber: {sumber}")
        tambal(sumber)

        # --no-build-isolation penting: tanpa itu pip membuat lingkungan
        # terpisah yang tidak memuat numpy, dan ./setup.py gagal membacanya.
        # --no-deps karena seluruh dependensi sudah didaftarkan di
        # requirements.txt dengan versi terkunci.
        print("  memasang paket ...")
        jalan(
            [argumen.pip, "-m", "pip", "install", "--no-build-isolation", "--no-deps", str(sumber)],
            "pemasangan selesai",
        )

        verifikasi(argumen.pip)
    finally:
        if milik_sendiri and not argumen.sisakan:
            shutil.rmtree(kerja, ignore_errors=True)

    print("=== Selesai ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
