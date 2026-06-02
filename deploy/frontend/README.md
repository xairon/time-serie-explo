# Déploiement du frontend JUNON (handoff DSI)

Déploiement du **frontend seul** (site web) sur le cluster Kubernetes de la DSI.
Le backend + GPU + base de données restent sur `dib-2019006065`.

## 1. L'architecture en une image

```
   Internet ──https://<hostname>──►  [ Front nginx + React ]   (cluster K8s DSI)
                                             │
                                             │  /api/  (réseau interne université)
                                             ▼
                                      [ Back FastAPI + GPU ]    (dib-2019006065 → 10.195.25.16:49514)
                                      [ PostgreSQL, MLflow ]
```

- Le navigateur ne parle **qu'au front**. Le front (nginx) **relaie `/api/`** vers le backend de dib.
- Le backend n'est **jamais** exposé à Internet (joignable seulement depuis le front, en interne).
- `/api/` et le site sont sur la même origine → **cookie de session** sans souci cross-site.

## 2. Fourni clé en main dans ce dépôt

Tout est prêt ; la DSI n'a en principe **rien à écrire**, juste à brancher.

| Élément | Fichier | Rôle |
|---|---|---|
| **CI build+push** | `.gitlab-ci.yml` (racine) | À chaque push `main` touchant le front → build de l'image + push sur le registry du projet (`$CI_REGISTRY_IMAGE/frontend:latest` + tag SHA). Runner **kaniko** (pas de Docker privilégié). |
| **Image** | `deploy/frontend/Dockerfile` | SPA React compilé + nginx qui sert le statique **et** proxie `/api/`. Contexte de build = racine du dépôt. |
| **Proxy backend** | `deploy/frontend/nginx.conf.template` | `${DIB_BACKEND}` substitué au démarrage (envsubst). |
| **Deployment** | `deploy/frontend/k8s/deployment.yaml` | 1 réplica, `DIB_BACKEND=10.195.25.16:49514`, probes, limites CPU/mémoire. |
| **Service** | `deploy/frontend/k8s/service.yaml` | ClusterIP interne (port 80). |
| **Ingress** | `deploy/frontend/k8s/ingress.yaml` | Entrée publique + TLS. **Seul placeholder à remplir : le `host`.** |

## 3. Le workflow (une fois branché)

```
push sur main  ──►  CI : build image frontend  ──►  push registry  ──►  K8s tire l'image  ──►  déploiement
```

C'est exactement le flux décrit par la DSI. Côté dév : on pousse du code, rien d'autre.

## 4. Ce qu'on attend de la DSI (le minimum)

1. 🔴 **Route réseau** : pods du cluster → `10.195.25.16:49514`. **Point critique** — sans ça, le
   site s'affiche mais aucune donnée ne remonte (502 sur `/api/`).
2. 🔴 **Hostname public + DNS** (ex. `junon.univ-tours.fr`) → à reporter dans `k8s/ingress.yaml`.
3. 🟡 **TLS** : ingress-controller nginx + cert-manager pour le certificat auto (sinon, certificat fourni à la main).
4. 🟢 **Registry/CI** : le job utilise les variables GitLab standard (`$CI_REGISTRY*`, token de job) — rien à configurer sauf si le runner kaniko doit être adapté à votre infra.

Réponses aux questions du ticket :
- **URL souhaitée** : à définir (voir point 2) — à reporter dans `ingress.yaml`.
- **Stockage persistant** : **non**. Le front est *stateless* (fichiers statiques + reverse-proxy) ; tout l'état est sur dib.
- **Flux sortants du front** : depuis le **pod**, uniquement → `10.195.25.16:49514` (backend) + DNS.
  Côté **navigateur** (client), l'app charge en plus les **tuiles carto** (`basemaps.cartocdn.com`) et
  les **Google Fonts** (`fonts.googleapis.com`, `fonts.gstatic.com`) — à autoriser si la politique
  réseau filtre aussi le poste client.

Côté `dib` : une fois le front en HTTPS, passer `COOKIE_SECURE=true` dans `deploy/dib-backend/.env` puis recréer le backend.

## 5. Réglages (les seuls)

| Fichier | Réglage | Valeur |
|---|---|---|
| `k8s/ingress.yaml` | `host:` + `tls.hosts` | le hostname public retenu |
| `k8s/deployment.yaml` | `DIB_BACKEND` | `10.195.25.16:49514` (déjà rempli) |
| `k8s/deployment.yaml` | `image:` | `$CI_REGISTRY_IMAGE/frontend:latest` (déjà rempli) |

## 6. Déployer / mettre à jour (côté DSI)

```bash
kubectl apply -f deploy/frontend/k8s/        # 1er déploiement
kubectl get pods,svc,ingress                 # vérifs
kubectl rollout restart deploy/junon-frontend # forcer une maj d'image
```

## 7. Vérifier

1. `https://<hostname>` → l'observatoire (public) s'affiche.
2. Connexion avec un compte → l'atelier répond (preuve que `/api/` atteint dib).
3. Site OK mais 502 sur `/api/` → c'est le **point 1 du §4** (route réseau vers dib).

## 8. Filet de sécurité

Le front tourne déjà sur `dib` en local (validation) — l'app reste utilisable en interne
pendant la mise en place K8s. Build/test sans registry :

```bash
docker build -f deploy/frontend/Dockerfile -t junon-frontend:test .
cd deploy/frontend && docker compose up -d --build   # front local, port 49513
```
