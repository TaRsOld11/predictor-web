# Sentiment Django — Uji Model Sentimen (TF-IDF + SVM)

Web 1 halaman berbasis **Django** untuk menguji model sentimen Bahasa Indonesia
hasil training (TF-IDF + SVM, dilabeli otomatis dengan IndoBERTweet di notebook training).

---

## Struktur Proyek

```
sentiment_django/
├── manage.py
├── requirements.txt
├── build.sh                   ← build command untuk Render
├── Procfile                   ← start command (gunicorn)
├── sentiment_django/          ← konfigurasi Django
│   ├── settings.py
│   └── urls.py
└── predictor/                 ← aplikasi utama
    ├── urls.py
    ├── views.py               ← preprocessing + logika prediksi
    ├── templates/
    │   └── predictor/
    │       └── index.html     ← tampilan halaman
    └── ml_model/               ← ← LETAKKAN FILE MODEL HASIL TRAINING DI SINI
        ├── README.txt
        ├── pipeline_sentimen.pkl    (belum ada, harus ditambahkan)
        ├── label_encoder.pkl        (belum ada, harus ditambahkan)
        ├── kamus_normalisasi.json   (belum ada, harus ditambahkan)
        └── stopwords.json           (belum ada, harus ditambahkan)
```

---

## 1. Pasang Dependencies (lokal)

```bash
pip install -r requirements.txt
```

> **Penting:** versi `scikit-learn` di `requirements.txt` harus sama (atau kompatibel)
> dengan versi yang dipakai saat training di Colab. Cek dengan
> `import sklearn; print(sklearn.__version__)` di Colab, lalu samakan di sini.

---

## 2. Letakkan File Model Hasil Training (Mendukung Multi-Model)

Web ini mendukung **lebih dari satu model sekaligus** — user bisa memilih
lewat dropdown di halaman. Setiap model ditaruh di subfolder sendiri di
`predictor/ml_model/<nama_folder>/`:

```
predictor/ml_model/
├── svm_tfidf_v1/
│   ├── pipeline_sentimen.pkl
│   ├── label_encoder.pkl
│   ├── kamus_normalisasi.json
│   ├── stopwords.json
│   └── config.json
└── model_lainnya/
    └── ... (struktur sama)
```

| File | Isi |
|---|---|
| `pipeline_sentimen.pkl` | sklearn Pipeline (TF-IDF + classifier), sudah di-fit |
| `label_encoder.pkl` | LabelEncoder untuk decode hasil prediksi |
| `kamus_normalisasi.json` | kamus normalisasi slang/alay (harus identik dgn training model ini) |
| `stopwords.json` | daftar stopword (harus identik dgn training model ini) |
| `config.json` | `{"nama_tampilan": "...", "deskripsi": "..."}` — teks yang muncul di dropdown |

**Cara menambah model baru:** cukup buat subfolder baru berisi 5 file di
atas, lalu restart server. Tidak perlu ubah kode apapun — folder baru
otomatis terdeteksi dan muncul di dropdown.

Folder yang belum lengkap (kurang salah satu file wajib) otomatis diabaikan
dari dropdown tanpa membuat error, selama masih ada model lain yang valid.
Kalau tidak ada satupun model valid, halaman menampilkan pesan error dan
form disembunyikan.

---

## 3. Jalankan Server (lokal)

```bash
python manage.py migrate
python manage.py runserver
```

Buka browser ke: **http://127.0.0.1:8000/**

---

## 4. Deploy ke Render (Free Tier)

1. Push seluruh project (termasuk folder `predictor/ml_model/` yang sudah berisi
   4 file model) ke repo GitHub.
2. Di Render: **New → Web Service** → connect repo.
3. **Build Command**: `bash build.sh`
4. **Start Command**: `gunicorn sentiment_django.wsgi:application`
   (atau biarkan Render mendeteksi otomatis dari `Procfile`)
5. **Environment Variables** yang disarankan:
   | Key | Value |
   |---|---|
   | `DEBUG` | `False` |
   | `SECRET_KEY` | (generate string random sendiri, jangan pakai default) |
   | `ALLOWED_HOSTS` | `localhost,127.0.0.1` (Render otomatis menambahkan hostname-nya sendiri) |
6. Tunggu build selesai, lalu akses URL yang diberikan Render.

> File model (`pipeline_sentimen.pkl` dkk) untuk TF-IDF+SVM biasanya kecil
> (puluhan KB–beberapa MB), jadi aman di-commit langsung ke Git tanpa Git LFS,
> kecuali ukurannya melebihi ~50MB.

---

## Cara Kerja

1. User pilih model dari dropdown, ketik teks di form → klik **Jalankan Prediksi**
2. Django POST ke view `index`
3. View memuat model + kamus + stopword dari subfolder model yang dipilih di
   `ml_model/` (di-cache per model, hanya load 1x per proses server, bukan
   per-request)
4. Teks diproses lewat 6 tahap (sama seperti notebook training):
   cleaning → case folding → tokenisasi → normalisasi → stopword removal → stemming
5. Teks bersih masuk ke pipeline TF-IDF + classifier model yang dipilih →
   prediksi label: **Positif / Netral / Negatif** + estimasi confidence
6. Hasil + teks bersih (untuk verifikasi) + confidence bar ditampilkan di halaman yang sama

## Info Model di Halaman

Di bawah dropdown, ada baris ringkas berisi:
- **Chip teknik balancing** yang dipakai model itu (atau badge "Baseline ·
  tanpa balancing" kalau tidak pakai teknik apapun)
- Link **"Info model"** di kanan — klik untuk expand panel berisi deskripsi,
  accuracy, F1-score, total data, serta detail precision/recall dan
  distribusi label

Panel ini default tertutup supaya tidak menutupi form input teks. Semua
datanya diambil dari `config.json` di folder model masing-masing (lihat
`predictor/ml_model/README.txt` untuk format lengkapnya).

---

## Catatan Performa

Stemming (Sastrawi) memproses kata satu per satu dan relatif lambat dibanding
tahap lain. Untuk teks pendek (1 kalimat, sesuai kebutuhan halaman testing ini),
ini tidak masalah. Kalau nanti dikembangkan untuk memproses banyak teks sekaligus
(misal upload CSV), pertimbangkan menjalankan stemming secara batch/async.
