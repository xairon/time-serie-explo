#!/bin/sh
# Applique les migrations Alembic au démarrage puis lance l'API.
# Évite les décalages schéma/code après un rebuild (cf. login cassé par
# 0002_auth_events non appliquée).
set -e

echo "[entrypoint] alembic upgrade head…"
alembic upgrade head

echo "[entrypoint] démarrage uvicorn…"
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
