# Déploiement du frontend JUNON

Guide pas-à-pas pour déployer **uniquement le frontend** (site web) sur le cluster
Kubernetes de la DSI, le backend + GPU + base de données restant sur `dib-2019006065`.

> Pas besoin de connaître Kubernetes pour suivre ce guide : les commandes sont
> données telles quelles. La section « Concepts » explique ce qu'on fait et pourquoi.

---

## 1. L'idée en une image

```
   Internet  ──https://<hostname>──►  [ Front nginx + React ]   (cluster K8s DSI)
                                              │
                                              │  /api/  (réseau interne université)
                                              ▼
                                       [ Back FastAPI + GPU ]    (dib-2019006065:49514)
                                       [ PostgreSQL, MLflow ]
```

- Le navigateur ne parle **qu'au front**. Le front (nginx) relaie `/api/` vers le back.
- Le back n'est **jamais** exposé à Internet : joignable seulement depuis le front, en interne.
- Comme `/api/` est servi sur la même URL que le site, le **cookie de connexion** marche
  sans souci de cross-site.

---

## 2. Concepts Kubernetes (le minimum)

| Terme | C'est quoi | Fichier ici |
|-------|-----------|-------------|
| **Image** | Une « boîte » figée contenant le site compilé + nginx | construite via `Dockerfile` |
| **Registry** | L'entrepôt où on dépose l'image pour que K8s la récupère | `registry.scm.univ-tours.fr` |
| **Deployment** | « Lance N copies de la boîte et surveille-les » | `k8s/deployment.yaml` |
| **Service** | Adresse interne stable vers les copies (non publique) | `k8s/service.yaml` |
| **Ingress** | La porte d'entrée publique + le HTTPS | `k8s/ingress.yaml` |
| **kubectl** | La télécommande pour piloter le cluster | (outil CLI) |
| **kubeconfig** | Tes identifiants + l'adresse du cluster (fournis par la DSI) | `~/.kube/config` |

---

## 3. Prérequis (à obtenir de la DSI)

Voir aussi `docs/reponse-ticket-106843.md`. Bloquants tant qu'ils manquent :

1. 🔴 **Route réseau** ouverte : pods du cluster → `10.195.25.16:49514` (sinon le site
   s'affiche mais aucune donnée ne remonte). **C'est LE point critique.**
2. 🔴 **Namespace** dédié + **kubeconfig** (pour que `kubectl` déploie).
3. 🔴 **Registry** d'images + identifiants `docker login`.
4. 🔴 **Hostname public** + entrée **DNS** (ex. `junon.univ-tours.fr`).
5. 🟡 **TLS** : confirmer qu'un ingress-controller **nginx** et **cert-manager** sont en
   place (pour le certificat HTTPS automatique). Sinon, fournir le certificat à la main.

Et côté `dib` : une fois le front en HTTPS, passer `COOKIE_SECURE=true` dans
`deploy/dib-backend/.env` puis recréer le backend.

---

## 4. Construire et pousser l'image

Depuis la **racine du dépôt** (le `.` final est important : c'est le contexte de build) :

```bash
# Build (remplace le tag par ton registry/projet)
docker build -f deploy/frontend/Dockerfile \
  -t registry.scm.univ-tours.fr/ringuet/junon-frontend:latest .

# Connexion au registry de la fac, puis envoi de l'image
docker login registry.scm.univ-tours.fr
docker push registry.scm.univ-tours.fr/ringuet/junon-frontend:latest
```

Pour juste **tester que ça build** sans registry :
```bash
docker build -f deploy/frontend/Dockerfile -t junon-frontend:test .
```

---

## 5. Renseigner les placeholders

Trois valeurs à remplacer avant d'appliquer :

| Fichier | Ligne | Remplacer |
|---------|-------|-----------|
| `k8s/deployment.yaml` | `image:` | par l'image poussée au §4 |
| `k8s/deployment.yaml` | `DIB_BACKEND` | doit valoir `10.195.25.16:49514` (déjà bon) |
| `k8s/ingress.yaml` | `host:` et `tls.hosts` | par le hostname fourni par la DSI |

---

## 6. Déployer

```bash
# Applique les 3 manifestes (Deployment + Service + Ingress)
kubectl apply -f deploy/frontend/k8s/

# Vérifs
kubectl get pods                 # le pod junon-frontend doit être "Running"
kubectl get svc                  # le Service ClusterIP existe
kubectl get ingress              # montre le hostname + l'adresse
kubectl logs deploy/junon-frontend   # logs nginx en cas de souci
```

Mettre à jour après un nouveau build :
```bash
docker build ... && docker push ...           # nouvelle image
kubectl rollout restart deploy/junon-frontend  # recharge l'image
kubectl rollout status  deploy/junon-frontend  # suit le redéploiement
```

---

## 7. Vérifier que tout marche

1. Ouvrir `https://<hostname>` → l'observatoire (public) doit s'afficher.
2. Se connecter avec un compte → l'atelier doit répondre (preuve que `/api/` atteint dib).
3. Si le site s'affiche mais que les données ne chargent pas / 502 sur `/api/` :
   c'est presque toujours le **point 1 du §3** (route réseau vers dib fermée).

---

## 8. Filet de sécurité

Le frontend tourne déjà sur `dib` en validation (`docker-compose.yml` de ce dossier,
port `49513`). Si le déploiement K8s coince, l'app reste utilisable en interne pendant
la coordination avec la DSI — le K8s est la version publique propre, pas un prérequis
pour que l'application fonctionne.

```bash
# Lancer/maj le front en local sur dib (validation)
cd deploy/frontend && docker compose up -d --build
```
