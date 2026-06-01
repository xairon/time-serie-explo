# Design — Système de comptes utilisateurs (JUNON)

> Date : 2026-06-01 · Statut : validé (en attente de relecture finale)

## 1. Objectif & contexte

Donner à chaque utilisateur un compte pour gérer **ses propres** ressources
d'atelier (datasets, modèles, scénarios), dans la perspective d'une exposition
publique de l'observatoire (accès BRGM).

Décisions cadrées avec l'utilisateur :

- **Comptes internes provisionnés par un admin** (pas d'auto-inscription, pas de SSO).
  Login **email / mot de passe**. Deux rôles : `admin`, `user`.
- **Observatoire public** (routers `observatory/*` accessibles sans login) ;
  **atelier privé** (datasets, training, models, forecasting, explainability,
  counterfactual, pastas) derrière login et cloisonné **par propriétaire**.
- **Strict privé** : un utilisateur ne voit/gère que ses ressources ; l'**admin
  voit tout**. Aucun partage entre utilisateurs en v1 (YAGNI).

### État actuel (exploration)

- **Aucune authentification** nulle part ; tous les endpoints sont ouverts.
- **Base Postgres `junon_db` vide** : moteur SQLAlchemy async présent
  (`api/database.py`, `get_db()` inutilisé), aucune table, aucun ORM, pas d'Alembic.
- **Modèles** → MLflow (id = `run_id`), métadonnées en tags MLflow, **pas de propriétaire**.
- **Datasets** → fichiers `data/prepared/{id}/` (`config.yaml` + `data.csv`), **pas de propriétaire**.
- **Scénarios** → artefacts MLflow rattachés à un `run_id`.
- CORS autorise déjà l'en-tête `Authorization`. Middleware = CORS uniquement.
- Front/back **même origine** côté navigateur (nginx front proxie `/api/`).

## 2. Approche retenue

**Approche 1** : session par **JWT court dans un cookie `httpOnly`** + propriété
stockée dans les **métadonnées natives** de chaque ressource (tag MLflow / `config.yaml`).
Postgres ne porte que les comptes. Pas de store de sessions séparé.

Stack standard : SQLAlchemy 2.0 async + **Alembic**, **`pwdlib[argon2]`** (hachage),
**PyJWT** (jetons).

## 3. Modèle de données

Table `users` (Postgres, SQLAlchemy `Base`, migration Alembic) :

| champ | type | note |
|---|---|---|
| `id` | UUID (PK) | identifiant propriétaire des ressources |
| `email` | str, unique, indexé | identifiant de login |
| `display_name` | str | affiché dans l'UI |
| `password_hash` | str | argon2 via `pwdlib` |
| `role` | enum `admin` / `user` | |
| `is_active` | bool (def. true) | `false` = blocage immédiat |
| `token_version` | int (def. 0) | incrément = déconnexion partout |
| `created_at` | timestamp | |
| `last_login_at` | timestamp, nullable | |

## 4. Authentification

### Mécanisme

- Login → JWT signé (HS256, secret en config/env) contenant `sub=user.id`,
  `tv=token_version`, `exp` (court, ex. 12 h). Posé dans un cookie
  **`httpOnly` + `Secure` + `SameSite=Strict`**.
- Dépendance `get_current_user` : décode le cookie, **recharge l'utilisateur en
  base**, vérifie `is_active` et que `tv` == `user.token_version`. Sinon 401.
- **Révocation sans store** : `is_active=false` ou `token_version++` invalide les
  cookies en cours (revérifiés à chaque requête).
- CSRF neutralisé par `SameSite=Strict` + même origine (pas de token CSRF en v1).

### Endpoints `api/routers/auth.py`

- `POST /api/v1/auth/login` `{email, password}` → pose le cookie, renvoie l'utilisateur (sans hash). Met à jour `last_login_at`.
- `POST /api/v1/auth/logout` → efface le cookie.
- `GET /api/v1/auth/me` → utilisateur courant.
- `POST /api/v1/auth/change-password` `{old_password, new_password}` → revérifie l'ancien, re-hache, `token_version++`.

### Endpoints admin `api/routers/admin.py` (require_admin)

- `GET /api/v1/admin/users` → liste.
- `POST /api/v1/admin/users` `{email, display_name, role, initial_password}` → crée.
- `PATCH /api/v1/admin/users/{id}` → `is_active`, `role`, ou réinitialisation de mot de passe (`token_version++`).
- Pas de suppression dure en v1 (désactivation).

### Dépendances `api/auth/deps.py`

- `get_current_user` → 401 si cookie absent/invalide/utilisateur inactif.
- `require_admin` → 403 si non-admin.

### Gating des routers (`api/main.py`)

- Gated via `dependencies=[Depends(get_current_user)]` :
  `datasets, training, models, forecasting, explainability, counterfactual, pastas`.
- **Inchangés / publics** : tous les `observatory*` routers.

### Bootstrap

- `scripts/create_admin.py` : crée le premier admin (interactif ou args).
  Pas de bootstrap par variables d'env en clair.

## 5. Propriété & enforcement

Le propriétaire est stocké dans les métadonnées **natives** de chaque ressource.

- **Modèles** : tag MLflow `owner_id` posé à la création du run (injection de
  `current_user` dans le pipeline d'entraînement / `mlflow_client`).
  - Liste : `search_runs(filter_string="tags.owner_id = '<id>'")` ; admin → sans filtre.
  - Accès / suppression / dérivés (forecast, explain, residuals…) : charger le run,
    vérifier `owner_id == user.id` ou `role==admin`, sinon **404** (ne pas divulguer l'existence).
- **Datasets** : `owner_id` écrit dans `config.yaml` à la création/import
  (`dashboard/utils/dataset_registry.py`). Liste filtrée par propriétaire ; accès/suppression vérifiés.
- **Scénarios** : rattachés à un `run_id` modèle → **héritent** du propriétaire du
  modèle. Contrôle d'accès via la propriété du modèle parent.
- **Forecast / explain / counterfactual** : opèrent sur un `model_id` et/ou
  `dataset_id` → enforcement par vérification de propriété de la ressource référencée.

Helper commun (ex. `api/auth/ownership.py`) : `assert_owns_model(user, run_id)`,
`assert_owns_dataset(user, dataset_id)` pour factoriser les checks (404 sinon).

## 6. Migration de l'existant

`scripts/assign_legacy_ownership.py` (one-shot) :

- Tous les runs MLflow sans tag `owner_id` → tag = id de l'admin bootstrap.
- Tous les datasets sans `owner_id` dans `config.yaml` → owner = admin bootstrap.
- Réassignation ultérieure : hors scope v1 (l'admin voit tout de toute façon).

## 7. Frontend

- **`AuthContext`** (React) : `GET /auth/me` au montage, expose `user / login / logout`.
  Toutes les requêtes API passent `credentials: 'include'` (cookie envoyé
  automatiquement, jamais lu en JS) — à câbler dans `frontend/src/lib/api.ts`.
- **Page `/login`** + menu compte (déconnexion, changement de mot de passe).
- **Gardes de routes** : pages atelier → redirection vers `/login` si non connecté ;
  routes observatoire **publiques**.
- **Listes "mes ressources"** : pas de changement d'UI — le backend filtre déjà par
  propriétaire. Sur 401, l'UI redirige vers `/login`.
- **Page admin** (gestion des comptes : créer, activer/désactiver, rôle, reset mdp),
  visible aux `admin` uniquement.

## 8. Gestion d'erreurs

- 401 (non authentifié) → le front redirige vers `/login`.
- 403 (authentifié mais non autorisé : endpoint admin) → message « accès refusé ».
- 404 sur ressource d'un autre propriétaire (pas de 403, pour ne pas divulguer l'existence).
- Échecs de login : message générique (« identifiants invalides ») sans distinguer
  email inconnu / mot de passe faux.

## 9. Tests

- **Backend (pytest)** :
  - Login OK / mauvais mot de passe / compte désactivé.
  - Révocation : un cookie devient invalide après `token_version++` / `is_active=false`.
  - Gating : atelier → 401 sans cookie ; observatoire → 200 sans cookie.
  - Isolation : user A ne voit pas et ne peut pas accéder (404) au modèle/dataset/
    scénario de user B ; l'admin y accède.
  - Admin : création/désactivation/reset par un admin ; refus (403) pour un `user`.
- **Front** : test minimal de garde de route + flux login → accès atelier.

## 10. Hors scope v1 (YAGNI)

- Auto-inscription, vérification email, reset par email.
- SSO université (CAS/LDAP/Shibboleth).
- Partage de ressources entre utilisateurs ; visibilité en lecture inter-utilisateurs.
- Réassignation fine de propriété ; suppression dure de comptes.
- Token CSRF dédié (couvert par SameSite + même origine).
