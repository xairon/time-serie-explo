# Environnement de développement isolé (sur dib)

Deux environnements **totalement séparés** tournent sur `dib-2019006065` :

| | **PROD** | **DEV** |
|---|---|---|
| Code | branche `main` | branche `dev` |
| Compose | `docker-compose.yml` (+ `.cuda.yml`) | `docker-compose.dev.yml` |
| Projet docker | `time-serie-explo` | `junon-dev` |
| Front | `:49513` (+ K8s public) | `:49518` |
| Backend API | `:49514` (← front K8s) | `:49516` |
| MLflow | `:49512` | `:49517` |
| Base PostgreSQL | volume `postgres_data` | volume `postgres_data_dev` |
| Modèles / artefacts | `mlflow_artifacts`, `data/`, `checkpoints/` | `mlflow_artifacts_dev`, `data-dev/`, `checkpoints-dev/` |

→ Un modèle entraîné, une migration ou un compte créé en **dev** n'affecte **jamais** la prod.

## Lancer / arrêter le dev

```bash
# Démarrer (GPU inclus dans le fichier → le -f explicite est correct ici)
docker compose -p junon-dev -f docker-compose.dev.yml up -d --build

# Arrêter (garde les données dev)
docker compose -p junon-dev -f docker-compose.dev.yml down

# Tout remettre à zéro (supprime aussi les volumes dev)
docker compose -p junon-dev -f docker-compose.dev.yml down -v
```

La base dev est vierge au premier démarrage : appliquer les migrations puis créer un admin.

```bash
docker exec junon-backend-dev sh -lc 'cd /app && alembic upgrade head'
docker exec junon-backend-dev sh -lc 'cd /app && python3 scripts/create_admin.py --email <toi@ex.fr> --name "<Nom>"'
```

## Workflow

1. Bosser sur `dev`, tester sur l'environnement dev (`:49518`).
2. Quand c'est bon : merge `dev` → `main`.
3. Le merge sur `main` reconstruit l'image front (CI) → la DSI redéploie ; côté dib,
   reconstruire la stack prod si le **backend** a changé :
   `docker compose up -d --build` (utilise le `COMPOSE_FILE` de `.env`, GPU préservé).

> ⚠️ Ne reconstruis **pas** la stack prod (`docker compose up`) depuis la branche `dev` :
> le backend prod `:49514` (utilisé par le front K8s) prendrait alors le code de dev.
