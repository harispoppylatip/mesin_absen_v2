"""Periksa layanan wajah InsightFace terhadap kontrak yang dipakai website.

Skrip ini berbicara ke layanan yang SUDAH jalan lewat HTTP, sama seperti yang
dilakukan aplikasi Laravel. Jadi bila skrip ini lulus, berarti kontraknya benar
benar berfungsi, bukan hanya kode yang terlihat benar.

    python verifikasi.py                          periksa dengan contoh bawaan
    python verifikasi.py --dir ..\faces           ambil contoh dari folder lain
    python verifikasi.py --url http://127.0.0.1:5005
    python verifikasi.py --benchmark              ukur waktu proses berulang
    python verifikasi.py --jarak 1.11             sekaligus uji ambang jarak

Hasil yang diharapkan: semua baris bertanda OK, dan jarak orang sama jauh di
bawah ambang sedangkan jarak orang berbeda jauh di atas ambang.
"""

from __future__ import annotations

import argparse
import base64
import io
import statistics
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import requests

EKSTENSI = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BAWAAN_URL = "http://127.0.0.1:5005"
BAWAAN_DIMENSI = 512

lulus = 0
gagal = 0


def periksa(syarat: bool, pesan: str, tambahan: str = "") -> bool:
    """Catat satu pemeriksaan."""
    global lulus, gagal

    if syarat:
        lulus += 1
        print(f"  OK   {pesan}" + (f"  {tambahan}" if tambahan else ""))
    else:
        gagal += 1
        print(f"  GAGAL {pesan}" + (f"  {tambahan}" if tambahan else ""))

    return syarat


def kirim_encode(url: str, berkas: Path, with_photo: bool = False, require_single: bool = False) -> dict:
    """Kirim satu foto ke /encode."""
    opsi = {"with_photo": "1"} if with_photo else {}
    if require_single:
        opsi["require_single"] = "1"
    with berkas.open("rb") as gagang:
        balasan = requests.post(
            f"{url}/encode",
            files={"image": (berkas.name, gagang, "image/jpeg")},
            params=opsi,
            timeout=120,
        )
    return {"http": balasan.status_code, "badan": balasan.json()}


def kirim_detect(url: str, berkas: Path, with_photo: bool = False) -> dict:
    """Kirim satu foto ke /detect."""
    opsi = {"with_photo": "1"} if with_photo else {}
    with berkas.open("rb") as gagang:
        balasan = requests.post(
            f"{url}/detect",
            files={"image": (berkas.name, gagang, "image/jpeg")},
            params=opsi,
            timeout=120,
        )
    return {"http": balasan.status_code, "badan": balasan.json()}


def jarak(angka_a: list[float], angka_b: list[float]) -> float:
    """Jarak euclid dua angka wajah. Angka sudah panjangnya 1 dari layanan."""
    a = np.asarray(angka_a, dtype=np.float32)
    b = np.asarray(angka_b, dtype=np.float32)

    return float(np.linalg.norm(a - b))


def kumpulkan(dir_foto: Path, url: str, batas_per_orang: int = 4) -> dict[str, list[dict]]:
    """Hitung angka wajah untuk beberapa foto tiap orang."""
    orang: dict[str, list[dict]] = {}

    for folder in sorted(p for p in dir_foto.iterdir() if p.is_dir()):
        for berkas in sorted(folder.iterdir()):
            if berkas.suffix.lower() not in EKSTENSI:
                continue
            if len(orang.get(folder.name, [])) >= batas_per_orang:
                break

            hasil = kirim_encode(url, berkas, require_single=True)
            badan = hasil["badan"]

            if hasil["http"] != 200 or badan.get("status") != "success":
                continue

            orang.setdefault(folder.name, []).append(
                {"nama": berkas.name, "angka": badan["encoding"], "sisi": badan["face_size"]}
            )

    return orang


def jalankan_benchmark(url: str, berkas: Path, ulang: int = 15) -> None:
    """Ukur waktu proses berulang."""
    print()
    print("=" * 74)
    print("BENCHMARK")
    print("=" * 74)

    for label, pemanggil in (("/encode", kirim_encode), ("/detect", kirim_detect)):
        # Sekali pemanasan supaya bukan biaya awal yang terukur.
        pemanggil(url, berkas)
        catatan = []

        for _ in range(ulang):
            import time

            mulai = time.perf_counter()
            hasil = pemanggil(url, berkas)
            catatan.append((time.perf_counter() - mulai) * 1000)

            if hasil["http"] != 200:
                print(f"  {label}: HTTP {hasil['http']}, benchmark dihentikan")
                return

        print(
            f"  {label:9s} median {statistics.median(catatan):7.1f} ms"
            f" (min {min(catatan):7.1f} maks {max(catatan):7.1f}) dari {ulang} kiriman"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Periksa layanan wajah InsightFace.")
    parser.add_argument("--url", default=BAWAAN_URL, help="alamat layanan")
    parser.add_argument("--dir", default=str(Path(__file__).resolve().parent.parent / "faces"), help="folder contoh foto")
    parser.add_argument("--foto", default=None, help="satu foto tertentu untuk uji kontrak")
    parser.add_argument("--jarak", type=float, default=0.0, help="uji ambang jarak, contoh 1.11")
    parser.add_argument("--benchmark", action="store_true", help="ukur waktu proses berulang")
    argumen = parser.parse_args()

    url = argumen.url.rstrip("/")
    dir_foto = Path(argumen.dir)

    print("=" * 74)
    print("PERIKSA LAYANAN WAJAH INSIGHTFACE")
    print("=" * 74)
    print(f"alamat : {url}")
    print(f"contoh : {dir_foto}")

    # ------------------------------------------------------------ kesehatan
    print()
    print("KESEHATAN")
    try:
        balasan = requests.get(f"{url}/health", timeout=30)
    except requests.RequestException as galat:
        print(f"  GAGAL layanan tidak bisa dihubungi: {galat}")
        print()
        print("  Jalankan dulu layanannya di jendela lain:")
        print("      .\\.venv\\Scripts\\python.exe app.py")

        return 1

    sehat = balasan.json()
    print(f"  HTTP {balasan.status_code}, status {sehat.get('status')}, model {sehat.get('model')}")

    periksa(balasan.status_code == 200, "layanan membalas 200 pada /health", f"HTTP {balasan.status_code}")
    periksa(sehat.get("model_ready") is True, "model siap", str(sehat.get("model_error")))
    periksa(
        str(sehat.get("dimensions")) == str(BAWAAN_DIMENSI),
        f"dimensi angka wajah {BAWAAN_DIMENSI}",
        f"terbaca {sehat.get('dimensions')}",
    )
    periksa(
        isinstance(sehat.get("model"), str) and sehat["model"] != "",
        "nama model dilaporkan",
        str(sehat.get("model")),
    )

    # ------------------------------------------------------ pilih contoh foto
    if argumen.foto:
        contoh = [Path(argumen.foto)]
    elif dir_foto.is_dir():
        semua = sorted(p for p in dir_foto.rglob("*") if p.is_file() and p.suffix.lower() in EKSTENSI)
        # Foto dengan wajah besar lebih meyakinkan untuk uji kontrak.
        contoh = [p for p in semua if p.stat().st_size > 40_000][:3] or semua[:3]
    else:
        contoh = []

    if not contoh:
        print()
        print(f"  Tidak ada foto contoh di {dir_foto}. Uji kontrak dilewati.")

        return 1 if gagal else 0

    # ------------------------------------------------------------ /encode
    print()
    print("ENCODE SATU ORANG")
    berkas = contoh[0]
    print(f"  foto: {berkas}")

    hasil = kirim_encode(url, berkas, with_photo=True)
    badan = hasil["badan"]
    print(f"  HTTP {hasil['http']}, status {badan.get('status')}, pesan {badan.get('message')}")

    periksa(hasil["http"] == 200, "HTTP 200", f"HTTP {hasil['http']}")
    periksa(badan.get("status") == "success", "status success", str(badan.get("status")))
    periksa(
        badan.get("model") == sehat.get("model"),
        "nama model di /encode sama dengan /health",
        str(badan.get("model")),
    )

    angka = badan.get("encoding")

    if periksa(isinstance(angka, list) and len(angka) == BAWAAN_DIMENSI, f"angka wajah {BAWAAN_DIMENSI} nilai", f"panjang {len(angka) if isinstance(angka, list) else 'bukan larik'}"):
        panjang = float(np.linalg.norm(np.asarray(angka, dtype=np.float32)))
        periksa(
            abs(panjang - 1.0) < 1e-3,
            "angka wajah sudah panjangnya 1 sehingga jarak bisa dibandingkan sebagai cosine",
            f"panjang {panjang:.6f}",
        )

    periksa(
        isinstance(badan.get("face_size"), int) and badan["face_size"] > 0,
        "ukuran wajah dilaporkan",
        f"{badan.get('face_size')} px",
    )
    periksa(
        isinstance(badan.get("face_count"), int) and badan["face_count"] >= 1,
        "jumlah wajah dilaporkan",
        str(badan.get("face_count")),
    )

    # -------------------------------------------------------------- foto bukti
    print()
    print("FOTO BUKTI")
    b64 = badan.get("photo")
    if periksa(isinstance(b64, str) and b64 != "", "foto bukti disertakan saat with_photo=1"):
        mentah = base64.b64decode(b64)
        periksa(mentah[:2] == b"\xff\xd8", "foto bukti berformat JPEG", mentah[:2].hex())
        periksa(
            badan.get("photo_bytes") == len(mentah),
            "jumlah bita yang dilaporkan cocok dengan isi berkasnya",
            f"{len(mentah)} bita",
        )
        periksa(
            len(mentah) <= 48 * 1024,
            "foto bukti tidak lebih dari 48 KB",
            f"{len(mentah) / 1024:.1f} KB pada mutu {badan.get('photo_quality')}",
        )
        periksa(
            max(badan.get("photo_width", 0), badan.get("photo_height", 0)) <= 480,
            "foto bukti diperkecil paling besar 480 px",
            f"{badan.get('photo_width')}x{badan.get('photo_height')}",
        )

    # ------------------------------------------------------------- tanpa bukti
    hasil = kirim_encode(url, berkas)
    periksa(
        "photo" not in hasil["badan"] or hasil["badan"].get("photo") is None,
        "foto bukti TIDAK disertakan bila tidak diminta",
    )

    # ------------------------------------------------------------------ /detect
    print()
    print("DETECT SELURUH WAJAH")

    # Cari gambar dengan wajah terbanyak supaya uji banyak wajah benar benar jalan.
    pilihan = None
    terbanyak = 0

    kandidat = contoh
    if dir_foto.is_dir():
        kandidat = sorted(
            (p for p in dir_foto.rglob("*") if p.is_file() and p.suffix.lower() in EKSTENSI),
            key=lambda p: -p.stat().st_size,
        )[:6]

    for berkas_uji in kandidat:
        hasil_uji = kirim_detect(url, berkas_uji, with_photo=True)
        jumlah = hasil_uji["badan"].get("face_count", 0)

        if jumlah > terbanyak:
            terbanyak = jumlah
            pilihan = hasil_uji

    if pilihan is None:
        periksa(False, "ada foto yang bisa diperiksa di /detect")
    else:
        badan = pilihan["badan"]
        print(f"  HTTP {pilihan['http']}, jumlah wajah {badan.get('face_count')}, bingkai {badan.get('width')}x{badan.get('height')}")
        periksa(pilihan["http"] == 200, "HTTP 200", f"HTTP {pilihan['http']}")
        periksa(badan.get("status") == "success", "status success", str(badan.get("status")))
        periksa(
            badan.get("face_count") == len(badan.get("faces", [])),
            "jumlah wajah cocok dengan panjang daftar wajah",
            f"{badan.get('face_count')} lawan {len(badan.get('faces', []))}",
        )

        lebar = badan.get("width", 0)
        tinggi = badan.get("height", 0)
        di_dalam = True
        urut = True
        sebelumnya = None
        angka_per_wajah = []

        for wajah in badan.get("faces", []):
            kotak = wajah.get("bbox", {})

            if not (
                0 <= kotak.get("left", -1) < lebar
                and 0 <= kotak.get("top", -1) < tinggi
                and 0 < kotak.get("right", lebar + 1) <= lebar
                and 0 < kotak.get("bottom", tinggi + 1) <= tinggi
                and kotak.get("right", 0) > kotak.get("left", 0)
                and kotak.get("bottom", 0) > kotak.get("top", 0)
            ):
                di_dalam = False

            luas = (kotak.get("right", 0) - kotak.get("left", 0)) * (kotak.get("bottom", 0) - kotak.get("top", 0))

            if sebelumnya is not None and luas > sebelumnya:
                urut = False

            sebelumnya = luas

            if isinstance(wajah.get("encoding"), list):
                angka_per_wajah.append(wajah)

        periksa(di_dalam, "semua kotak wajah berada di dalam bingkai dan ukurannya masuk akal")
        periksa(urut, "wajah terbesar dilaporkan paling dulu")
        periksa(len(angka_per_wajah) == badan.get("face_count"), "setiap wajah punya angka wajah")

        if terbanyak >= 2:
            print(f"  foto dengan {terbanyak} wajah ikut terperiksa")

    # ------------------------------------------------- daya pisah dan ambang
    if dir_foto.is_dir() and not argumen.foto:
        print()
        print("DAYA PISAH SKOR")
        orang = kumpulkan(dir_foto, url)

        datar = [
            (nama, item["angka"], item["sisi"])
            for nama, daftar in orang.items()
            for item in daftar
        ]
        print(f"  {len(datar)} foto dari {len(orang)} orang dihitung: {', '.join(f'{n} {len(d)}' for n, d in orang.items())}")

        sama = []
        beda = []

        for (nama_a, angka_a, _), (nama_b, angka_b, _) in combinations(datar, 2):
            nilai = jarak(angka_a, angka_b)
            (sama if nama_a == nama_b else beda).append(nilai)

        if sama and beda:
            def ke_cosine(nilai: float) -> float:
                """Padanan cosine dari jarak, untuk dibandingkan dengan catatan lama."""
                return 1.0 - (nilai * nilai) / 2.0

            print(
                f"  orang sama   : n={len(sama)} min {min(sama):.4f} maks {max(sama):.4f} rata {statistics.mean(sama):.4f}"
                f"  (cosine {ke_cosine(max(sama)):+.4f} sampai {ke_cosine(min(sama)):+.4f})"
            )
            print(
                f"  orang beda   : n={len(beda)} min {min(beda):.4f} maks {max(beda):.4f} rata {statistics.mean(beda):.4f}"
                f"  (cosine {ke_cosine(max(beda)):+.4f} sampai {ke_cosine(min(beda)):+.4f})"
            )

            # Jarak kecil berarti mirip. Jadi celah aman ditentukan pasangan
            # orang sama yang PALING BESAR jaraknya (paling tidak mirip)
            # dibandingkan pasangan orang berbeda yang PALING KECIL jaraknya
            # (paling mirip). Selama kedua nilai itu belum bertemu, ambang di
            # antaranya memisahkan keduanya dengan bersih.
            celah_aman = min(beda) - max(sama)
            periksa(
                celah_aman > 0,
                "tidak ada orang berbeda yang lebih mirip daripada orang sama",
                f"celah {celah_aman:+.4f} (orang sama paling jauh {max(sama):.4f},"
                f" orang beda paling mirip {min(beda):.4f})",
            )

            if celah_aman > 0:
                print(
                    f"  ambang yang aman berada di antara {max(sama):.4f} dan {min(beda):.4f}"
                    f"  (tengah {max(sama) + celah_aman / 2:.4f})"
                )
            if argumen.jarak > 0:
                salah_terima = sum(1 for nilai in beda if nilai <= argumen.jarak)
                salah_tolak = sum(1 for nilai in sama if nilai > argumen.jarak)
                print(
                    f"  ambang {argumen.jarak:.4f}: orang lain lolos {salah_terima}/{len(beda)},"
                    f" orang sama ditolak {salah_tolak}/{len(sama)}"
                )
                periksa(salah_terima == 0, f"ambang {argumen.jarak} menolak seluruh orang lain")
        else:
            print("  pasangan tidak cukup untuk mengukur daya pisah")

    if argumen.benchmark:
        jalankan_benchmark(url, berkas)

    print()
    print("=" * 74)
    print(f"HASIL: {lulus} lulus, {gagal} gagal")
    print("=" * 74)

    return 1 if gagal else 0


if __name__ == "__main__":
    raise SystemExit(main())
