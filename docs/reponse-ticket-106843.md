# Réponse — Ticket n°106843 (hébergement JUNON)

> Brouillon à relire avant envoi. Proposer un sous-domaine ou laisser la DSI choisir.

---

Bonjour Olivier,

Merci pour les précisions.

Après vérification, le GPU est nécessaire à l'usage courant (pas seulement à
l'entraînement) : l'application entraîne et exécute ses propres modèles. Comme le GPU
et l'entrepôt de données sont tous deux sur `dib-2019006065`, je propose de **séparer
l'application en deux** :

- **Chez vous (Kubernetes) : seulement le frontend.** Un conteneur nginx léger qui sert
  l'interface web et relaie les appels API vers le backend. Pas de GPU, ressources
  faibles. C'est le seul élément exposé au public.
- **Sur `dib-2019006065` : le backend + l'entrepôt + le GPU.** Non exposé à Internet :
  joignable uniquement par le frontend, sur le réseau interne (`10.195.25.16`, port `49514`).

**Vos deux questions :**

1. *Entrepôt de données* : pas besoin de le redéployer, il reste sur `dib-2019006065`.
2. *Hébergement K8s* : oui, et c'est même simplifié — sans GPU à héberger de votre
   côté, ce n'est plus bloquant.

**Ce dont j'aurais besoin de votre côté :**

- Que le frontend (sur votre cluster) puisse joindre `10.195.25.16:49514` en interne.
- Vos modalités habituelles pour : déposer une image de conteneur, déployer sur le
  cluster, et exposer un nom de domaine en HTTPS.

Je fournis le conteneur du frontend et des manifestes Kubernetes (Deployment / Service /
Ingress) que j'adapterai à vos contraintes. Dites-moi comment vous procédez d'habitude
et ce qu'il vous faut de ma part.

Cordialement,
Nicolas Ringuet
Ingénieur de recherche — LIFAT, Université de Tours
Projet ANR JUNON
