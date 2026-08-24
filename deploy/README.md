# Separate frontend / backend deployment

Target architecture (Option A — reverse proxy): the **frontend** (React + nginx) is
hosted remotely (Kubernetes, IT department (DSI)) and is the **only public entry
point**. It proxies `/api/` to the **backend** (FastAPI + GPU + Postgres warehouse),
which stays on `<backend-host>`, accessible only via the **university internal
network**.

```
[BRGM browser, external]
      │ HTTPS (public, handled by the K8s Ingress)
      ▼
[K8s — DSI]  junon-frontend (nginx: static SPA + /api/ proxy)
      │ /api/  → university internal network (port BACKEND_PORT)
      ▼
[<backend-host>]  FastAPI backend + Postgres + Redis + MLflow + GPU (RTX A6000)
```

The frontend calls the API using a relative path (`API_BASE = '/api/v1'`), so it is
the **same origin** from the browser's perspective: no CORS, no changes to the
frontend code, and the backend is **never exposed to the Internet**.

## 1. Backend on the on-premise host

```bash
cd deploy/dib-backend
cp .env.example .env          # mots de passe + ALLOWED_ORIGINS (URL publique du front)
BACKEND=cuda docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

- Exposes the API on `BACKEND_PORT` (default 49514) — **internal network only**.
- Keeps the GPU (training + inference), Postgres, Redis, and MLflow co-located.
- `ALLOWED_ORIGINS` must contain the public URL served by the frontend.

## 2. Frontend (Kubernetes, DSI)

Build and push the image (context = repository root):

```bash
docker build -f deploy/frontend/Dockerfile -t registry.scm.univ-tours.fr/ringuet/junon-frontend:latest .
docker push registry.scm.univ-tours.fr/ringuet/junon-frontend:latest
```

Adapt and then apply the manifests (image, `DIB_BACKEND`, TLS hostname):

```bash
kubectl apply -f deploy/frontend/k8s/
```

### Variant without K8s (dedicated server/VM)

```bash
cd deploy/frontend
cp .env.example .env          # DIB_BACKEND=<ip-ou-dns-dib>:49514
docker compose up -d --build
```

## Network prerequisites

- **Inbound**: public access to the frontend (K8s Ingress, or port 49513 otherwise).
- **Internal**: the frontend (K8s) must be able to reach `<backend-host>:BACKEND_PORT`.
  → this is the equivalent of the DB port already opened on the university network.
- **Outbound Internet**: from the **backend** (dib) for Hub'Eau; from the
  **browser** for the map tiles. The K8s frontend only needs to be able to reach dib.
