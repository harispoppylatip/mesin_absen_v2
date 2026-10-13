# Layanan Wajah InsightFace

Mesin pengenal wajah untuk aplikasi absen solat SMKIM4. Layanan ini berdiri
sendiri sebagai layanan HTTP kecil, dan aplikasi Laravel menghubunginya untuk
mengubah foto wajah menjadi angka pembanding.

```
Laravel  ---POST /encode--->  Layanan ini  --->  512 angka wajah
         <---POST /detect---               --->  angka + kotak wajah
```

## Kenapa dipisah dari aplikasi PHP

Memuat model wajah memerlukan beberapa detik dan ratusan megabita memori. Di
dalam PHP, biaya itu dibayar ulang pada setiap permintaan sehingga satu absen
memakan belasan detik. Di layanan ini model dimuat SEKALI, lalu dipakai
berulang untuk seluruh permintaan.

Ini layanan pengganti untuk mesin dlib yang dipakai sebelumnya. Angka wajah
dari kedua mesin TIDAK bisa dibandingkan, sebab panjang dan skalanya berbeda.

## Isi folder

| Berkas                  | Gunanya                                                       |
| ----------------------- | ------------------------------------------------------------- |
| `app.py`                | Seluruh layanan: model, aturan, dan dua endpoint HTTP         |
| `tambal_insightface.py` | Memasang insightface tanpa compiler C++                       |
| `verifikasi.py`         | Memeriksa layanan yang sedang jalan terhadap kontrak aplikasi |
| `requirements.txt`      | Pustaka beserta versi terkunci                                |
| `Dockerfile`            | Citra dua tahap: pembangun lalu layanan                       |
| `docker-compose.yml`    | Menjalankan sebagai wadah, tersambung ke aplikasi web         |
| `.env.example`          | Seluruh pengaturan yang bisa diubah beserta penjelasannya     |

## Menjalankan di komputer (tanpa Docker)

```powershell
cd C:\Users\Pongo\Documents\IoT\instighface

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r absen-solat\requirements.txt
.\.venv\Scripts\python.exe absen-solat\tambal_insightface.py

# Model diunduh sekali, sekitar 16 MB untuk buffalo_sc.
.\.venv\Scripts\python.exe absen-solat\app.py --preload

# Layanan jalan di http://127.0.0.1:5005
.\.venv\Scripts\python.exe absen-solat\app.py
```

Bila model sudah ada di folder lain, arahkan lewat `FACE_MODEL_ROOT`. Contoh:
`$env:FACE_MODEL_ROOT="C:\Users\Pongo\Documents\IoT\instighface\cache\models"`.

## Memeriksa bahwa layanan benar

```powershell
# Jendela pertama: jalankan layanannya
.\.venv\Scripts\python.exe absen-solat\app.py

# Jendela kedua: periksa
.\.venv\Scripts\python.exe absen-solat\verifikasi.py --dir .\faces --jarak 1.15 --benchmark
```

Skrip itu memeriksa kesehatan layanan, panjang dan normalisasi angka wajah,
foto bukti, kotak wajah, urutan wajah, serta daya pisah skor antara orang yang
sama dan orang yang berbeda. Semua baris harus bertanda OK.

## Menjalankan dengan Docker

```powershell
docker network create absen-jaringan     # sekali saja, sebelum pertama
cd C:\Users\Pongo\Documents\IoT\instighface\absen-solat
docker compose build
docker compose up -d
docker compose logs -f wajah
```

Dua hal yang perlu diketahui tentang cara ini:

1. Nama layanan sengaja `wajah`, sebab nama itu sudah dipakai aplikasi Laravel
   sebagai `FACE_SERVICE_URL_DOCKER=http://wajah:5005`. Jadi tidak ada yang
   perlu diubah di sisi aplikasi.
2. Container ikut ke jaringan bersama `absen-jaringan` dengan alias `wajah`.
   Bila aplikasi web juga berjalan sebagai container, keduanya langsung saling
   melihat tanpa membuka port apa pun ke komputer.

Port dibuka ke `127.0.0.1` saja secara bawaan, karena layanan ini tidak punya
kata sandi. Alamat itu diperlukan bila aplikasi Laravel dijalankan langsung di
komputer dengan `php artisan serve`.

Untuk membukanya ke komputer lain di jaringan sekolah, ubah `FACE_PUBLISH_HOST`
di berkas `.env` menjadi `0.0.0.0`, dan sadari bahwa siapa pun yang bisa
menghubungi port itu bisa memakai layanan ini.

## Kontrak yang dipakai aplikasi Laravel

| Permintaan     | Balasan                                                                 |
| -------------- | ----------------------------------------------------------------------- |
| `GET /health`  | Kesiapan model. Membalas 503 selama model belum termuat                 |
| `POST /encode` | Angka wajah untuk SATU orang, foto terbesar dipakai                     |
| `POST /detect` | Angka wajah untuk SELURUH wajah di bingkai, beserta kotak masing masing |

Keduanya menerima berkas pada field bernama `image`, dan menerima kueri
`with_photo=1` untuk meminta foto bukti serta `require_single=1` (khusus
`/encode`) untuk menolak foto yang memuat lebih dari satu wajah.

Status yang mungkin dibalas: `success`, `no_face`, `multiple_faces`,
`face_too_small` (dari aplikasi, berdasarkan `face_size`), `error`, dan
`service_unavailable` (di sisi aplikasi, saat layanan tidak bisa dihubungi).

Balasan `/encode`:

```json
{
  "status": "success",
  "encoding": [512 angka],
  "face_count": 1,
  "face_size": 559,
  "model": "insightface-buffalo_sc",
  "message": "Wajah terbaca.",
  "photo": "base64 JPEG",
  "photo_bytes": 24016,
  "photo_width": 360,
  "photo_height": 480,
  "photo_quality": 72
}
```

Balasan `/detect` memuat `faces`, masing masing berisi `index`, `bbox`
(`top`, `right`, `bottom`, `left` dalam piksel gambar ASLI), `face_size`,
`encoding` (bisa `null`), dan `error`.

## Tiga hal yang mudah salah dipahami

**Angka wajah tidak ternormalisasi sebelum dikirim.** Nilai mentah InsightFace
panjangnya sekitar 27. Layanan ini membaginya dengan panjangnya lebih dulu,
sehingga setiap angka yang keluar panjangnya 1. Karena itu jarak euclid antar
dua angka setara dengan cosine similarity, dan ambang jarak di aplikasi tetap
bisa dibaca sebagai kemiripan.

**Wajah yang angkanya gagal dihitung tetap dilaporkan.** Pada `/detect`, entri
seperti itu keluar dengan `encoding: null` dan `error` berisi alasannya. Ini
disengaja supaya satu wajah yang gagal tidak menggagalkan wajah lain, dan
halaman kamera tetap bisa menggambar kotaknya.

**Nama model ikut diuji aplikasi.** Aplikasi menolak angka wajah bila nama
model yang dilaporkan berbeda dari `FACE_MODEL` di berkas `.env`-nya. Ini
pengaman, bukan gangguan: angka dari dua model berbeda tidak setara, jadi
mencampurnya akan menghasilkan pengenalan yang salah. Bila nama model di sini
diubah, ubah juga di sisi aplikasi.

## Mengatur sumber daya

Waktu proses dan memori sudah diukur, jadi angkanya bisa dipilih dengan alasan.

**Waktu proses.** Diukur pada i7-13620H, satu wajah, median 30 kali:

| Setelan                        | Waktu                                      |
| ------------------------------ | ------------------------------------------ |
| sisi panjang detektor 640      | 21,1 ms                                    |
| sisi panjang detektor 320      | 4,7 ms                                     |
| bentuk input ikut rasio gambar | 2 kali lebih cepat daripada kanvas persegi |
| thread intra 1                 | 42,9 ms                                    |
| thread intra 2                 | 26,6 ms                                    |
| thread intra 4                 | 21,1 ms                                    |
| thread intra 8                 | 27,9 ms                                    |

Ditambah pengiriman HTTP dan penyusunan balasan, satu foto sampai diterima
aplikasi memerlukan sekitar 44 milidetik. Mesin dlib sebelumnya memerlukan 12
sampai 15 detik untuk foto yang sama.

Kesimpulan yang dipakai sebagai bawaan: `FACE_DETECT_LONG_SIDE=640`,
`FACE_DETECT_MATCH_ASPECT=1`, dan `FACE_ORT_INTRA_THREADS=4`.

**Jumlah thread sebaiknya sama dengan jumlah core yang tersedia.** Menambah
thread melebihi jumlah core justru memperlambat, seperti terlihat pada angka
intra 8 di atas. Di dalam Docker, samakan `FACE_ORT_INTRA_THREADS` dengan
`FACE_CPUS`. Pada server dua core, pakai 2 untuk keduanya.

**Memori.** Setelah melayani lebih dari 50 permintaan dengan buffalo_sc, proses
layanan memakai 242 MB dan puncaknya 256 MB. Karena itu batas wadah dipasang
768 MB, sekitar tiga kali puncak terukur. Memakai buffalo_l memerlukan batas
yang lebih besar, sebab bobot modelnya saja 326 MB.

Menurunkan `FACE_DETECT_LONG_SIDE` ke 320 menghemat sekitar empat kali waktu,
dengan risiko wajah kecil atau orang yang jauh dari kamera tidak terbaca.
Ukur dulu dengan `verifikasi.py --benchmark` sebelum menurunkannya.

## Memilih model

| Model      | Isi                      | Bobot   | Satu wajah |
| ---------- | ------------------------ | ------- | ---------- |
| buffalo_sc | det_500m + MobileFaceNet | ~16 MB  | 20,2 ms    |
| buffalo_l  | det_10g + ResNet50       | ~326 MB | 168,7 ms   |

buffalo_sc dipakai sebagai bawaan. Pengukuran atas 34 foto milik 5 orang
menunjukkan celah amannya masih lebar: orang berbeda berada pada jarak 1,1650
sampai 1,5032, sedangkan orang sama berada di bawah 1,1427. Ambang 1,15
diletakkan di antara keduanya.

Ganti model hanya bila ada bukti bahwa buffalo_sc salah mengenali orang di
sekolah. Mengganti model membuat seluruh angka wajah yang tersimpan tidak
terpakai lagi, jadi seluruh anggota harus mendaftar ulang wajahnya.

## Berkas terkait

- `C:\Users\Pongo\Documents\laravel\absen_solat` - aplikasi Laravel-nya
- `C:\Users\Pongo\Documents\laravel\absen_solat\README.md` - panduan sisi aplikasi, termasuk bagian Mesin pengenal wajah
- `C:\Users\Pongo\Documents\laravel\absen_solat\README-docker.md` - panduan Docker sisi aplikasi
- `C:\Users\Pongo\Documents\IoT\instighface` - folder induk yang memuat skrip pengujian kamera dan foto contoh
