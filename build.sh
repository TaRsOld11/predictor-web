#!/usr/bin/env bash
# build.sh — set sebagai Build Command di Render

set -o errexit

pip install -r requirements.txt

# Download data NLTK yang dibutuhkan tokenizer (sekali saat build, bukan saat request)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

python manage.py collectstatic --no-input
python manage.py migrate
