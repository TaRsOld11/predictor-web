STRUKTUR MULTI-MODEL
=====================

Setiap model ditaruh di SUBFOLDER-nya sendiri di dalam ml_model/, bukan
langsung di sini. Halaman web akan otomatis mendeteksi semua subfolder
yang lengkap dan menampilkannya sebagai pilihan dropdown.

Contoh struktur:

  predictor/ml_model/
  ├── svm_baseline/
  │   ├── pipeline_sentimen.pkl
  │   ├── label_encoder.pkl
  │   ├── kamus_normalisasi.json
  │   ├── stopwords.json
  │   └── config.json
  ├── svm_filter_smote/
  │   └── ... (4 file wajib + config.json)
  └── model_lainnya/
      └── ... (4 file wajib + config.json)

CARA MENAMBAH MODEL BARU
=========================
1. Buat subfolder baru di ml_model/, nama bebas (huruf/angka/underscore,
   tanpa spasi), misal: "svm_undersampling".
2. Dari notebook training model itu, copy 4 file wajib ke dalamnya:
     - pipeline_sentimen.pkl   (sklearn Pipeline: TF-IDF + classifier, sudah fit)
     - label_encoder.pkl       (LabelEncoder hasil training)
     - kamus_normalisasi.json  (kamus slang/alay yang dipakai SAAT training model ini)
     - stopwords.json          (daftar stopword yang dipakai SAAT training model ini)
3. Buat file config.json di folder yang sama. Semua field di bawah ini
   OPSIONAL — kalau tidak diisi, bagian terkait di halaman web otomatis
   disembunyikan (tidak akan error).

     {
       "nama_tampilan": "SVM + TF-IDF (Filter Confidence + SMOTE)",
       "deskripsi": "Data difilter berdasarkan confidence score IndoBERTweet, label netral ambigu dihapus, lalu SMOTE diterapkan pada data training.",

       "teknik_balancing": ["Filter Confidence", "Hapus Netral Ambigu", "SMOTE"],

       "accuracy": 0.8654,
       "precision_weighted": 0.8630,
       "recall_weighted": 0.8654,
       "f1_weighted": 0.8640,

       "data_info": {
         "total": 410,
         "train": 287,
         "test": 123,
         "distribusi_label": {"positif": 180, "netral": 90, "negatif": 140}
       }
     }

   Catatan field "teknik_balancing":
     - Array kosong [] = ditandai sebagai BASELINE (badge abu-abu
       "Baseline · tanpa balancing").
     - Array berisi 1 atau lebih string = ditampilkan sebagai chip per
       teknik, contoh teknik yang relevan dengan eksperimen ini:
         "Filter Confidence"      -> filter data training berdasar
                                      confidence score IndoBERTweet
         "Hapus Netral Ambigu"    -> buang data label netral yang skornya
                                      tidak meyakinkan
         "Oversampling"
         "Undersampling"
         "SMOTE"
     - Bisa kombinasi lebih dari satu teknik dalam satu model, cukup
       tulis semuanya dalam array.

   Nilai accuracy/precision/recall/f1 ambil dari metadata.json yang
   sudah dihasilkan notebook Cell 17 (field "evaluasi"), tinggal copy
   angkanya (format desimal 0-1, akan otomatis dikonversi ke % di web).

4. Restart server — folder baru otomatis muncul di dropdown tanpa perlu
   ubah kode.

TAMPILAN DI WEB
================
- Dropdown untuk pilih model.
- Baris ringkas di bawah dropdown: chip teknik balancing (atau badge
  baseline) + link kecil "Info model" di kanan.
- Klik "Info model" untuk expand panel: deskripsi, accuracy, F1-score,
  total data, lalu detail precision/recall/distribusi label.
  Defaultnya tertutup supaya tidak menutupi form input teks.

CATATAN PENTING
================
- Tiap model punya kamus_normalisasi.json dan stopwords.json SENDIRI, karena
  preprocessing yang dipakai saat training bisa berbeda antar eksperimen.
  Jangan asumsikan satu kamus dipakai untuk semua model.
- Folder yang TIDAK lengkap (kurang salah satu dari 4 file wajib) akan
  diabaikan secara diam-diam dari dropdown — tidak akan muncul, tapi juga
  tidak akan membuat web error, selama masih ada minimal 1 folder lain yang
  lengkap.
- Kalau SEMUA folder tidak lengkap (tidak ada satupun model valid), halaman
  akan menampilkan pesan error dan form prediksi disembunyikan sampai
  minimal satu model siap.
