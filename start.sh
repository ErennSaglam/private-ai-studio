#!/usr/bin/env bash
# Private AI Studio - start the API (port 3001) and the web interface (port 3000) together.
#
# Usage (from the project folder):
#   ./start.sh                     base model
#   ./start.sh law-qa-r8-best      base model + the LoRA adapter in adapters/law-qa-r8-best
# Stop both with Ctrl+C.
set -euo pipefail

cd "$(dirname "$0")"
ADAPTER_NAME="${1:-}"

[ -x backend/.venv/bin/uvicorn ] || { echo "Kurulum eksik. Önce ./setup.sh komutunu çalıştırın."; exit 1; }
[ -d frontend/node_modules ] || { echo "Arayüz paketleri eksik. Önce ./setup.sh komutunu çalıştırın."; exit 1; }
if [ -n "$ADAPTER_NAME" ] && [ ! -f "adapters/$ADAPTER_NAME/adapters.safetensors" ]; then
  echo "adapters/$ADAPTER_NAME klasöründe adapter bulunamadı."
  exit 1
fi
for port in 3000 3001; do
  if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
    echo "$port numaralı port kullanımda. Önce o programı kapatın:  lsof -nP -iTCP:$port -sTCP:LISTEN"
    exit 1
  fi
done

echo "API başlatılıyor (model: Llama 3.2 1B${ADAPTER_NAME:+ + adapter: $ADAPTER_NAME})..."
(
  cd backend
  ADAPTER="$ADAPTER_NAME" HF_HUB_OFFLINE=1 exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 3001
) &
API_PID=$!
trap 'echo; echo "Kapatılıyor..."; kill $API_PID 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 60); do
  curl -s http://127.0.0.1:3001/api/health >/dev/null 2>&1 && break
  kill -0 $API_PID 2>/dev/null || { echo "API başlatılamadı. Yukarıdaki hata mesajına bakın."; exit 1; }
  sleep 1
done
echo "API hazır: http://localhost:3001/docs"
echo "Arayüz başlatılıyor: http://localhost:3000   (kapatmak için Ctrl+C)"
cd frontend && npm run dev
