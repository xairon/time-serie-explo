# Gestion des comptes et des secrets — JUNON

Procédé opérationnel pour les comptes utilisateurs (provisionnés par l'admin) et
pour les secrets d'infrastructure. À tenir à jour.

---

## Principe

- **Comptes internes** : pas d'auto-inscription, pas de SSO. L'admin crée chaque compte.
- **Rôles** : `admin` (voit tout, gère les comptes) et `user` (ne voit que ses propres
  datasets / modèles / scénarios — cloisonnement par `owner_id`).
- **Auth** : JWT en cookie httpOnly `junon_session`. Révocable côté serveur via
  `is_active` et `token_version` (rechargés en base à chaque requête).
- **Mots de passe** : hashés en **argon2** (`pwdlib`). **Personne ne les détient en clair**,
  pas même l'admin. Ne jamais tenir de liste des mots de passe utilisateurs.

---

## Créer un compte utilisateur

1. Se connecter en admin sur l'interface → menu compte → **Utilisateurs**
   (ou via l'API, voir plus bas).
2. Choisir email + nom affiché + rôle, et un **mot de passe temporaire** (min 8 car.,
   ex. `openssl rand -base64 12`).
3. Transmettre ce temporaire **une seule fois, par un canal sûr** : en personne, ou
   messagerie institutionnelle. **Éviter le mail en clair.**
4. Demander à l'utilisateur de **le changer dès la première connexion**
   (menu compte → changer le mot de passe). L'admin oublie alors le temporaire.

> ⚠️ L'email doit être un domaine réel : `email-validator` **refuse** `.local`, `.test`, etc.

### Via l'API (équivalent)

```bash
# 1. login admin -> cookie dans admin.jar
curl -s -c admin.jar -X POST http://<BACKEND>/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@exemple.fr","password":"<MDP_ADMIN>"}'

# 2. créer le compte (initial_password = temporaire)
curl -s -b admin.jar -X POST http://<BACKEND>/api/v1/admin/users \
  -H 'Content-Type: application/json' \
  -d '{"email":"nouveau@exemple.fr","display_name":"Prénom Nom","role":"user","initial_password":"<TEMP>"}'
```

Le script `scripts/verify_compartmentalization.py` automatise création + vérification
du cloisonnement (utile pour un compte de test).

---

## Réinitialiser un mot de passe (oubli)

L'admin **régénère un temporaire** — il ne peut pas récupérer l'ancien (hashé).

- Interface : page Utilisateurs → réinitialiser.
- API : `PATCH /api/v1/admin/users/{user_id}` avec `{"new_password":"<TEMP>"}`.

Effet de bord : `token_version` est incrémenté → toutes les sessions ouvertes de cet
utilisateur sont **immédiatement invalidées**. Idem pour désactivation (`is_active=false`)
et changement de rôle.

---

## Désactiver / révoquer un compte

`PATCH /api/v1/admin/users/{user_id}` avec `{"is_active": false}`.
La session en cours est coupée à la requête suivante (rechargement `is_active`).
On **désactive**, on ne supprime pas (préserve l'attribution `owner_id` des données).

---

## Secrets d'infrastructure → coffre

Contrairement aux mots de passe utilisateurs, ces secrets **doivent être conservés**
(dans un gestionnaire de mots de passe d'équipe : Bitwarden/Vaultwarden, KeePassXC,
ou le coffre de la DSI). Ne jamais les committer.

| Secret | Où il vit | Usage |
|--------|-----------|-------|
| `JWT_SECRET` | `deploy/dib-backend/.env` (gitignoré, sur dib) | Signature des JWT |
| `POSTGRES_PASSWORD` | idem | Base interne `junon_db` |
| `BRGM_DB_PASSWORD` | idem | Lecture entrepôt BRGM (gold) |
| Compte admin bootstrap | coffre d'équipe | Accès admin de secours |

- Le `.env` sur dib reste le point d'exécution, mais une **copie de référence** des
  secrets doit être dans le coffre (sinon perte du serveur = perte des secrets).
- `COOKIE_SECURE=false` sur dib (accès HTTP). En prod TLS (K8s DSI) → `true`.
- **Migration K8s (front DSI)** : les secrets côté cluster iront dans des **K8s Secrets**.
  Le `JWT_SECRET` reste sur dib (backend), non concerné par le cluster.

---

## Compte admin de secours

Bootstrap initial via `scripts/create_admin.py`. Garder au moins **un** compte admin
actif et ses identifiants dans le coffre. Si le dernier admin est perdu, recréer un
admin avec ce script directement sur dib.
