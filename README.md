# Sentiment Django — Uji Model Sentimen (TF-IDF + SVM)

Web 1 halaman berbasis **Django** untuk menguji model sentimen Bahasa Indonesia
hasil training (TF-IDF + SVM, dilabeli otomatis dengan IndoBERTweet di notebook training).

---

## Struktur Proyek

```
sentiment_django/
├── manage.py
├── requirements.txt
├── build.sh                   ← build command
├── Procfile
├── sentiment_django/          ← konfigurasi Django
│   ├── settings.py
│   └── urls.py
└── predictor/                 ← aplikasi utama
    ├── urls.py
    ├── views.py               ← preprocessing + logika prediksi
    ├── templates/
    │   └── predictor/
    │       └── index.html     ← tampilan halaman
    └── ml_model/
	├── svm_baseline/
	│   ├── pipeline_sentimen.pkl
	│   ├── label_encoder.pkl
	│   ├── kamus_normalisasi.json
	│   ├── stopwords.json
	│   └── config.json
	└── model_lainnya/
    	└── ... (struktur sama)
```

Detail file pada model:

| File | Isi |
|---|---|
| `pipeline_sentimen.pkl` | sklearn Pipeline (TF-IDF + classifier), sudah di-fit |
| `label_encoder.pkl` | LabelEncoder untuk decode hasil prediksi |
| `kamus_normalisasi.json` | kamus normalisasi slang/alay (harus identik dgn training model ini) |
| `stopwords.json` | daftar stopword (harus identik dgn training model ini) |
| `config.json` | `{"nama_tampilan": "...", "deskripsi": "..."}` — teks yang muncul di dropdown |

---

## 1. Pasang Dependencies (lokal)

```bash
pip install -r requirements.txt
```

---

## 2. Jalankan Server (lokal)

```bash
python manage.py migrate
python manage.py runserver
```

Buka browser ke: **http://127.0.0.1:8000/**

---

## Cara Kerja

1. User pilih model dari dropdown, ketik teks di form → klik **Jalankan Prediksi**
2. Django POST ke view `index`
3. View memuat model + kamus + stopword dari subfolder model yang dipilih di
   `ml_model/`
4. Teks diproses lewat 6 tahap:
   cleaning → case folding → tokenisasi → normalisasi → stopword removal → stemming
5. Teks bersih masuk ke pipeline TF-IDF + classifier model yang dipilih →
   prediksi label: **Positif / Netral / Negatif** + estimasi confidence
6. Hasil + teks bersih (untuk verifikasi) + confidence bar ditampilkan di halaman yang sama

