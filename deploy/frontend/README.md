# Déploiement du frontend JUNON (K8s)

Le **frontend** (SPA React + nginx) tourne sur le cluster Kubernetes ; il sert le site
**et** relaie `/api/` vers le **backend** resté sur `dib-2019006065` (`10.195.25.16:49514`).
Le backend n'est jamais exposé : seul le front l'appelle, en interne.

## Fourni dans le dépôt

| Fichier | Rôle |
|---|---|
| `.gitlab-ci.yml` (racine) | Build + push de l'image (`$CI_REGISTRY_IMAGE/frontend:latest`) à chaque push sur `main`. Runner kaniko. |
| `deploy/frontend/Dockerfile` | Image : SPA compilé + nginx (proxy `/api/`). Contexte de build = racine du dépôt. |
| `deploy/frontend/nginx.conf.template` | `${DIB_BACKEND}` substitué au démarrage. |
| `deploy/frontend/k8s/` | `deployment.yaml`, `service.yaml`, `ingress.yaml`. |

## À ajuster (les seuls réglages)

| Fichier | Réglage | Défaut |
|---|---|---|
| `k8s/ingress.yaml` | `host` + `tls.hosts` | `junon.univ-tours.fr` (à confirmer) |
| `k8s/deployment.yaml` | `DIB_BACKEND` | `10.195.25.16:49514` |
| `k8s/deployment.yaml` | `image` | `$CI_REGISTRY_IMAGE/frontend:latest` |

## Déployer

```bash
kubectl apply -f deploy/frontend/k8s/
kubectl get pods,svc,ingress
kubectl rollout restart deploy/junon-frontend   # forcer une maj d'image
```

## ⚠️ Le seul point bloquant côté réseau

Ouvrir la route **pods du cluster → `10.195.25.16:49514`**. Sans elle, le site s'affiche
mais `/api/` renvoie 502 (aucune donnée). Le frontend est *stateless* (aucun volume requis).

Test local de l'image sans registry : `docker build -f deploy/frontend/Dockerfile -t junon-frontend:test .`
