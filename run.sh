#!/usr/bin/env bash
# ./run.sh [up|down|logs|test]
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-up}" in
  down) docker compose down; exit 0 ;;
  logs) docker compose logs -f api; exit 0 ;;
  test) docker compose exec api pytest -q; exit 0 ;;
  up) ;;
  *) echo "uso: ./run.sh [up|down|logs|test]"; exit 1 ;;
esac

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Creado .env, completar las claves y volver a correr."
  exit 1
fi

for var in OPENROUTER_API_KEY PINECONE_API_KEY; do
  grep -Eq "^${var}=.+" .env || { echo "Falta ${var} en .env"; exit 1; }
done

docker compose up --build -d

echo -n "esperando API "
for _ in $(seq 1 90); do
  if curl -fs http://localhost:8000/health >/dev/null 2>&1; then
    echo
    echo "API:     http://localhost:8000"
    echo "Swagger: http://localhost:8000/docs"
    exit 0
  fi
  echo -n "."; sleep 2
done
echo
echo "la API no respondió: docker compose logs ingest api"
exit 1
