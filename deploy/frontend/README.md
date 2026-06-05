# JUNON frontend deployment (K8s)

The **frontend** (React SPA + nginx) runs on the Kubernetes cluster; it serves the site
**and** proxies `/api/` to the **backend**, which remains on `dib-2019006065` (`10.195.25.16:49514`).
The backend is never exposed: only the frontend calls it, internally.

## Provided in the repository

| File | Role |
|---|---|
| `.gitlab-ci.yml` (root) | Build + push of the image (`$CI_REGISTRY_IMAGE/frontend:latest`) on every push to `main`. kaniko runner. |
| `deploy/frontend/Dockerfile` | Image: compiled SPA + nginx (`/api/` proxy). Build context = repository root. |
| `deploy/frontend/nginx.conf.template` | `${DIB_BACKEND}` substituted at startup. |
| `deploy/frontend/k8s/` | `deployment.yaml`, `service.yaml`, `ingress.yaml`. |

## To adjust (the only settings)

| File | Setting | Default |
|---|---|---|
| `k8s/ingress.yaml` | `host` + `tls.hosts` | `junon.univ-tours.fr` (to be confirmed) |
| `k8s/deployment.yaml` | `DIB_BACKEND` | `10.195.25.16:49514` |
| `k8s/deployment.yaml` | `image` | `$CI_REGISTRY_IMAGE/frontend:latest` |

## Deploy

```bash
kubectl apply -f deploy/frontend/k8s/
kubectl get pods,svc,ingress
kubectl rollout restart deploy/junon-frontend   # force an image update
```

## ⚠️ The only blocking point on the network side

Open the route **cluster pods → `10.195.25.16:49514`**. Without it, the site displays
but `/api/` returns 502 (no data). The frontend is *stateless* (no volume required).

Local test of the image without a registry: `docker build -f deploy/frontend/Dockerfile -t junon-frontend:test .`
