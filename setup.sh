#!/usr/bin/env bash
# Private AI Studio - one-command setup for a fresh Apple Silicon Mac.
# Safe to run more than once: every step checks what is already installed and skips it.
#
# Usage (from the project folder):   ./setup.sh
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
MODEL_REPO="mlx-community/Llama-3.2-1B-Instruct-4bit"
MODEL_DIR="models/Llama-3.2-1B-Instruct-4bit"

step() { printf "\n\033[1;34m==> %s\033[0m\n" "$1"; }
ok()   { printf "    \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "    \033[31m✗ %s\033[0m\n" "$1"; exit 1; }

step "1/7 Bilgisayar kontrol ediliyor"
[ "$(uname -s)" = "Darwin" ] || fail "Bu proje yalnızca macOS üzerinde çalışır."
[ "$(uname -m)" = "arm64" ] || fail "Apple Silicon (M1, M2, M3, M4) işlemci gerekiyor. Intel işlemcili Mac'lerde MLX çalışmaz."
ok "macOS $(sw_vers -productVersion), Apple Silicon"
FREE_GB=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
[ "$FREE_GB" -ge 3 ] || fail "En az 3 GB boş disk alanı gerekiyor (şu an ${FREE_GB} GB)."
ok "Boş disk alanı: ${FREE_GB} GB"

step "2/7 Homebrew kontrol ediliyor"
if ! command -v brew >/dev/null 2>&1; then
  if [ -x /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  else
    echo "    Homebrew kurulu değil. Önce şu komutu çalıştırın, sonra ./setup.sh komutunu tekrar çalıştırın:"
    echo '    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    exit 1
  fi
fi
ok "Homebrew $(brew --version | head -1 | awk '{print $2}')"

step "3/7 Python 3.12 ve Node.js kontrol ediliyor"
if ! command -v python3.12 >/dev/null 2>&1; then
  echo "    Python 3.12 kuruluyor..."
  brew install python@3.12
fi
ok "$(python3.12 --version)"
if ! command -v node >/dev/null 2>&1; then
  echo "    Node.js kuruluyor..."
  brew install node
fi
ok "Node.js $(node --version)"

step "4/7 Python ortamı ve paketler hazırlanıyor"
if [ ! -x backend/.venv/bin/python ]; then
  python3.12 -m venv backend/.venv
  ok "backend/.venv oluşturuldu"
fi
VENV_VERSION=$(backend/.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
[ "$VENV_VERSION" = "3.12" ] || fail "backend/.venv Python $VENV_VERSION ile oluşturulmuş. Klasörü silip (rm -rf backend/.venv) ./setup.sh komutunu tekrar çalıştırın."
backend/.venv/bin/pip install --quiet --upgrade pip
backend/.venv/bin/pip install --quiet -r backend/requirements.txt
backend/.venv/bin/pip install --quiet -r training/requirements.txt
ok "Python paketleri kuruldu: $(backend/.venv/bin/pip list 2>/dev/null | grep -E '^mlx-lm ' | tr -s ' ')"

step "5/7 Model indiriliyor (yaklaşık 0.7 GB, yalnızca ilk kurulumda)"
if [ -f "$MODEL_DIR/model.safetensors" ]; then
  ok "Model zaten indirilmiş: $MODEL_DIR"
else
  backend/.venv/bin/hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
  ok "Model indirildi: $MODEL_DIR"
fi

step "6/7 Web arayüzü paketleri kuruluyor"
(cd frontend && npm install --silent)
ok "frontend/node_modules hazır"

step "7/7 Model internetsiz modda deneniyor"
HF_HUB_OFFLINE=1 backend/.venv/bin/mlx_lm.generate \
  --model "$MODEL_DIR" \
  --prompt "Türkiye'nin başkenti neresidir? Tek kelimeyle cevap ver." \
  --max-tokens 20 --temp 0 2>/dev/null | grep -vE "^=+$" | sed 's/^/    /'
ok "Model çalışıyor"

printf "\n\033[1;32mKurulum tamamlandı.\033[0m\n"
echo "Uygulamayı başlatmak için:                 ./start.sh"
echo "Hukuk verisiyle eğitilmiş modelle başlatmak için: ./start.sh law-qa-r8-best"
