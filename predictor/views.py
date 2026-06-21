import os
import re
import json
import joblib
import numpy as np
from django.conf import settings
from django.shortcuts import render

import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import emoji

# ---------------------------------------------------------------------------
# MULTI-MODEL REGISTRY
#
# Setiap model punya foldernya sendiri di predictor/ml_model/<nama_folder>/,
# berisi 4 file wajib + config.json. Folder baru otomatis terdeteksi tanpa
# perlu ubah kode — cukup tambahkan foldernya.
#
# Struktur tiap folder model:
#   predictor/ml_model/<nama_folder>/
#       ├── pipeline_sentimen.pkl
#       ├── label_encoder.pkl
#       ├── kamus_normalisasi.json
#       ├── stopwords.json
#       └── config.json   -> {"nama_tampilan": "...", "deskripsi": "..."}
# ---------------------------------------------------------------------------
ML_MODEL_ROOT = os.path.join(settings.BASE_DIR, 'predictor', 'ml_model')

REQUIRED_FILES = [
    'pipeline_sentimen.pkl',
    'label_encoder.pkl',
    'kamus_normalisasi.json',
    'stopwords.json',
]

LABEL_DISPLAY = {
    'positif': 'Positif', 'negatif': 'Negatif', 'netral': 'Netral',
}

# Cache di level module: setiap proses server load model maksimal sekali
# per model, lalu disimpan di sini.
_model_cache = {}
_stemmer_cache = {'instance': None}

# ---------------------------------------------------------------------------
# Contoh tweet untuk fitur "Acak Contoh"
#
# Sumber data: predictor/data/contoh_tweet.json, berisi list of object
# dengan minimal field "text". Field lain (tweet_url, id) diabaikan di sini,
# hanya teksnya yang dipakai untuk mengisi textarea secara acak.
# ---------------------------------------------------------------------------
CONTOH_TWEET_PATH = os.path.join(
    settings.BASE_DIR, 'predictor', 'data', 'contoh_tweet.json'
)


def list_contoh_tweet_text():
    """Baca predictor/data/contoh_tweet.json dan kembalikan list teks saja.
    Dibaca ulang setiap kali dipanggil (bukan di-cache) supaya perubahan
    isi file langsung muncul tanpa perlu restart server.
    """
    teks_list = []
    if os.path.exists(CONTOH_TWEET_PATH):
        try:
            with open(CONTOH_TWEET_PATH, encoding='utf-8') as f:
                data = json.load(f)
            teks_list = [
                item['text'].strip()
                for item in data
                if isinstance(item, dict) and item.get('text', '').strip()
            ]
        except Exception:
            teks_list = []

    return teks_list


def _ensure_nltk_data():
    for pkg in ('punkt', 'punkt_tab'):
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)


def _get_stemmer():
    # Stemmer Sastrawi tidak menyimpan state spesifik-model, aman dipakai
    # bersama oleh semua model -> cukup satu instance untuk semua.
    if _stemmer_cache['instance'] is None:
        _stemmer_cache['instance'] = StemmerFactory().create_stemmer()
    return _stemmer_cache['instance']


def list_available_models():
    """
    Scan folder ml_model/ dan kembalikan daftar model yang valid (4 file
    wajib lengkap). Dipanggil tiap request (murah, cuma os.listdir),
    supaya kalau ada folder model baru ditambahkan tanpa restart server
    pun langsung muncul di dropdown.

    Field dari config.json yang dibaca (semua opsional, ada default aman):
      - nama_tampilan      (str)  nama yang muncul di dropdown
      - deskripsi          (str)  catatan singkat tentang model ini
      - teknik_balancing    (list[str]) daftar teknik balancing data yang
                             dipakai. Array kosong [] = baseline (tanpa
                             balancing sama sekali).
      - accuracy, precision_weighted, recall_weighted, f1_weighted (float)
      - data_info: {total, train, test, distribusi_label: {...}}
    """
    hasil = []
    if not os.path.isdir(ML_MODEL_ROOT):
        return hasil

    for nama_folder in sorted(os.listdir(ML_MODEL_ROOT)):
        folder_path = os.path.join(ML_MODEL_ROOT, nama_folder)
        if not os.path.isdir(folder_path):
            continue

        missing = [f for f in REQUIRED_FILES
                   if not os.path.exists(os.path.join(folder_path, f))]
        if missing:
            continue  # folder belum lengkap, skip diam-diam dari dropdown

        config_path = os.path.join(folder_path, 'config.json')
        cfg = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, encoding='utf-8') as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}

        teknik_balancing = cfg.get('teknik_balancing', [])
        if not isinstance(teknik_balancing, list):
            teknik_balancing = [str(teknik_balancing)] if teknik_balancing else []

        is_baseline = len(teknik_balancing) == 0

        def _fmt_pct(val):
            try:
                return round(float(val) * 100, 1)
            except (TypeError, ValueError):
                return None

        data_info = cfg.get('data_info') or {}

        hasil.append({
            'id': nama_folder,
            'nama_tampilan': cfg.get('nama_tampilan', nama_folder),
            'deskripsi': cfg.get('deskripsi', ''),
            'teknik_balancing': teknik_balancing,
            'is_baseline': is_baseline,
            'accuracy': _fmt_pct(cfg.get('accuracy')),
            'precision_weighted': _fmt_pct(cfg.get('precision_weighted')),
            'recall_weighted': _fmt_pct(cfg.get('recall_weighted')),
            'f1_weighted': _fmt_pct(cfg.get('f1_weighted')),
            'data_total': data_info.get('total'),
            'data_train': data_info.get('train'),
            'data_test': data_info.get('test'),
            'distribusi_label': data_info.get('distribusi_label') or {},
        })

    return hasil


def _load_model_artifacts(model_id):
    """Load 4 artefak untuk satu model_id, dengan cache per model_id."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    folder_path = os.path.join(ML_MODEL_ROOT, model_id)
    missing = [f for f in REQUIRED_FILES
               if not os.path.exists(os.path.join(folder_path, f))]
    if not os.path.isdir(folder_path) or missing:
        raise FileNotFoundError(
            f"Model '{model_id}' tidak ditemukan atau belum lengkap. "
            f"Belum ada: {', '.join(missing) if missing else 'folder tidak ada'}."
        )

    _ensure_nltk_data()

    pipeline = joblib.load(os.path.join(folder_path, 'pipeline_sentimen.pkl'))
    label_encoder = joblib.load(os.path.join(folder_path, 'label_encoder.pkl'))

    with open(os.path.join(folder_path, 'kamus_normalisasi.json'), encoding='utf-8') as f:
        kamus_normalisasi = json.load(f)

    with open(os.path.join(folder_path, 'stopwords.json'), encoding='utf-8') as f:
        stopwords_id = set(json.load(f))

    artifacts = {
        'pipeline': pipeline,
        'label_encoder': label_encoder,
        'kamus_normalisasi': kamus_normalisasi,
        'stopwords_id': stopwords_id,
    }
    _model_cache[model_id] = artifacts
    return artifacts


# ---------------------------------------------------------------------------
# Tahap preprocessing — HARUS identik dengan notebook training masing-masing
# model. Kamus & stopword diambil per-model (artifacts), bukan global.
# ---------------------------------------------------------------------------
def _cleaning(text):
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'_', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _normalisasi(tokens, kamus):
    hasil = []
    for kata in tokens:
        kata = re.sub(r'(.)\1{2,}', r'\1\1', kata)
        kata = re.sub(r'(.)\1+', r'\1', kata)
        kata = kamus.get(kata, kata)
        if kata.strip():
            hasil.append(kata.strip())
    return hasil


def _hapus_stopword(tokens, stopwords_id):
    return [k for k in tokens if k not in stopwords_id and len(k) > 1]


def preprocess_teks(text, artifacts):
    text = _cleaning(text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = _normalisasi(tokens, artifacts['kamus_normalisasi'])
    tokens = _hapus_stopword(tokens, artifacts['stopwords_id'])
    stemmer = _get_stemmer()
    tokens = [stemmer.stem(k) for k in tokens]
    return ' '.join(tokens)


def predict_sentiment(text, model_id):
    """Mengembalikan (label_tampilan, confidence_persen_atau_None, teks_bersih)."""
    artifacts = _load_model_artifacts(model_id)

    teks_bersih = preprocess_teks(text, artifacts)

    if not teks_bersih.strip():
        return 'Netral', None, teks_bersih

    pipeline = artifacts['pipeline']
    le = artifacts['label_encoder']

    pred_encoded = pipeline.predict([teks_bersih])[0]
    raw_label = le.inverse_transform([pred_encoded])[0]
    label = LABEL_DISPLAY.get(raw_label, str(raw_label).capitalize())

    confidence = None
    try:
        dec = pipeline.decision_function([teks_bersih])[0]
        dec_exp = np.exp(dec - dec.max())
        proba_est = dec_exp / dec_exp.sum()
        confidence = round(float(proba_est.max()) * 100, 1)
    except Exception:
        pass

    return label, confidence, teks_bersih


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------
def index(request):
    available_models = list_available_models()
    context = {
        'available_models': available_models,
        'contoh_tweet_list': list_contoh_tweet_text(),
    }

    # Tentukan model yang dipilih: dari POST kalau ada, kalau tidak pakai
    # model pertama yang tersedia sebagai default.
    default_model_id = available_models[0]['id'] if available_models else None
    selected_model_id = request.POST.get('model_id', default_model_id)
    context['selected_model_id'] = selected_model_id

    if not available_models:
        context['error'] = (
            "Belum ada model yang siap dipakai di predictor/ml_model/. "
            "Tambahkan minimal satu folder model lengkap (lihat README.txt "
            "di folder tersebut)."
        )
        return render(request, 'predictor/index.html', context)

    if request.method == 'POST':
        text_input = request.POST.get('text_input', '').strip()
        context['text_input'] = text_input
        is_model_switch = request.POST.get('action') == 'switch_model'

        if is_model_switch:
            # Ganti dropdown model = cuma mau lihat info model yang baru
            # (chip balancing, accuracy, dll). Jangan jalankan prediksi apa
            # pun isi teksnya — biarkan textarea tetap terisi, tapi prediksi
            # baru jalan kalau user benar-benar klik "Jalankan Prediksi".
            pass
        elif not text_input:
            context['error'] = 'Teks tidak boleh kosong. Masukkan kalimat terlebih dahulu.'
        elif not selected_model_id or selected_model_id not in [m['id'] for m in available_models]:
            context['error'] = 'Model yang dipilih tidak valid. Coba pilih ulang dari daftar.'
        else:
            try:
                label, confidence, teks_bersih = predict_sentiment(text_input, selected_model_id)
                context['result'] = label
                context['confidence'] = confidence
                context['teks_bersih'] = teks_bersih
            except FileNotFoundError as e:
                context['error'] = str(e)
            except Exception as e:
                context['error'] = f'Terjadi error saat menjalankan prediksi: {e}'

    return render(request, 'predictor/index.html', context)
