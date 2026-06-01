# Réponse — Ticket n°106843 (hébergement JUNON)

> Brouillon à relire/ajuster avant envoi. Renseigner l'IP de `dib-2019006065`
> (Olivier l'a explicitement demandée) et le hostname public souhaité.

---

Bonjour Olivier,

Merci pour ces précisions, elles nous aident à clarifier l'architecture.

Après réflexion, le GPU est nécessaire **à l'usage courant** (pas seulement à
l'entraînement) : l'application entraîne et exécute des modèles de prévision
spécifiques (NBEATS, TFT, LSTM… via PyTorch/Darts). Ce ne sont pas des modèles
pré-entraînés servis par un moteur d'inférence générique ; les moteurs d'inférence
distants ne couvrent donc pas ce besoin. Le GPU et l'entrepôt de données étant tous
deux sur `dib-2019006065`, le plus simple et le plus sûr est de **séparer
l'application en deux** :

**1. Sur votre plateforme Kubernetes : uniquement le frontend.**
Un conteneur nginx léger qui sert l'interface React (statique) et relaie les appels
API vers le backend. C'est le **seul élément exposé à l'extérieur** (point d'entrée
public pour les collaborateurs BRGM). Pas de GPU requis, ressources très faibles
(~100 mCPU / 64–256 Mio).

**2. Sur `dib-2019006065` : le backend + l'entrepôt + le GPU.**
L'API FastAPI, PostgreSQL, Redis, MLflow et le GPU (RTX A6000) restent en place et
co-localisés. Le backend n'est **pas exposé sur Internet** : il est joignable
uniquement depuis le frontend K8s, sur le **réseau interne universitaire**.

Concrètement, le nginx du frontend proxie `/api/` vers
`dib-2019006065:49514` en interne ; côté navigateur, tout passe par l'URL publique
du frontend (même origine, donc pas de problématique CORS).

## Réponses à vos deux questions

**1. Entrepôt de données** — Pas besoin de le redéployer. Il reste sur
`dib-2019006065`, accès direct depuis K8s via le réseau interne. C'est le **même
besoin réseau** que celui déjà évoqué pour la base PostgreSQL (port ouvert sur le
réseau universitaire), simplement appliqué à l'API.
IP de `dib-2019006065` (réseau universitaire) : `10.195.25.16`, API exposée sur le
port `49514`.

**2. Hébergement K8s** — Oui, c'est toujours possible et même simplifié : comme
seul le frontend (sans GPU) est hébergé chez vous, l'absence de GPU dans vos clusters
n'est plus bloquante. Le GPU reste sur `dib-2019006065`.

## Besoins réseau pour le frontend sur K8s

- **Entrant** : accès public (Ingress + TLS) pour les collaborateurs BRGM.
- **Interne** : le pod frontend doit pouvoir joindre `dib-2019006065` sur le port de
  l'API (réseau universitaire).
- **Sortant Internet** : non requis pour le frontend lui-même (les appels Hub'Eau
  partent du backend sur dib ; les tuiles cartographiques sont chargées par le
  navigateur du client).

Je vous fournis le `docker-compose` du frontend et, si utile, des manifestes
Kubernetes prêts à adapter (Deployment / Service / Ingress).

Je reste disponible pour un échange.

Cordialement,
Nicolas Ringuet
Ingénieur de recherche — LIFAT, Université de Tours
Projet ANR JUNON
