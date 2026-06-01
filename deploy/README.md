# Déploiement séparé front / back

Architecture cible (Option A — reverse-proxy) : le **frontend** (React + nginx) est
hébergé à distance (Kubernetes DSI) et constitue le **seul point public**. Il proxie
`/api/` vers le **backend** (FastAPI + GPU + entrepôt Postgres) qui reste sur
`dib-2019006065`, accessible uniquement via le **réseau interne universitaire**.

```
[Navigateur BRGM, externe]
      │ HTTPS (public, géré par l'Ingress K8s)
      ▼
[K8s — DSI]  junon-frontend (nginx : SPA statique + proxy /api/)
      │ /api/  → réseau interne universitaire (port BACKEND_PORT)
      ▼
[dib-2019006065]  backend FastAPI + Postgres + Redis + MLflow + GPU (RTX A6000)
```

Le frontend appelle l'API en chemin relatif (`API_BASE = '/api/v1'`), donc **même
origine** côté navigateur : pas de CORS, pas de modification du code front, et le
backend n'est **jamais exposé à Internet**.

## 1. Backend sur dib-2019006065

```bash
cd deploy/dib-backend
cp .env.example .env          # mots de passe + ALLOWED_ORIGINS (URL publique du front)
BACKEND=cuda docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

- Expose l'API sur `BACKEND_PORT` (défaut 49514) — **réseau interne uniquement**.
- Garde le GPU (entraînement + inférence), Postgres, Redis, MLflow co-localisés.
- `ALLOWED_ORIGINS` doit contenir l'URL publique servie par le front.

## 2. Frontend (Kubernetes DSI)

Construire et pousser l'image (contexte = racine du dépôt) :

```bash
docker build -f deploy/frontend/Dockerfile -t registry.scm.univ-tours.fr/ringuet/junon-frontend:latest .
docker push registry.scm.univ-tours.fr/ringuet/junon-frontend:latest
```

Adapter puis appliquer les manifestes (image, `DIB_BACKEND`, hostname TLS) :

```bash
kubectl apply -f deploy/frontend/k8s/
```

### Variante hors K8s (serveur/VM dédié)

```bash
cd deploy/frontend
cp .env.example .env          # DIB_BACKEND=<ip-ou-dns-dib>:49514
docker compose up -d --build
```

## Pré-requis réseau

- **Entrant** : accès public au front (Ingress K8s / port 49513 sinon).
- **Interne** : le front (K8s) doit joindre `dib-2019006065:BACKEND_PORT`.
  → c'est l'équivalent du port DB déjà ouvert sur le réseau universitaire.
- **Sortant Internet** : depuis le **backend** (dib) pour Hub'Eau ; depuis le
  **navigateur** pour les tuiles cartographiques. Le front K8s n'a besoin que de
  joindre dib.
