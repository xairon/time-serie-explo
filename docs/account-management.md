# Account and Secret Management — JUNON

Operational procedure for user accounts (provisioned by the admin) and for
infrastructure secrets. Keep up to date.

---

## Principle

- **Internal accounts**: no self-registration, no SSO. The admin creates each account.
- **Roles**: `admin` (sees everything, manages accounts) and `user` (sees only their own
  datasets / models / scenarios — partitioning by `owner_id`).
- **Auth**: JWT in an httpOnly cookie `junon_session`. Revocable server-side via
  `is_active` and `token_version` (reloaded from the database on every request).
- **Passwords**: hashed with **argon2** (`pwdlib`). **Nobody holds them in cleartext**,
  not even the admin. Never keep a list of user passwords.

---

## Create a user account

1. Log in as admin in the interface → account menu → **Users**
   (or via the API, see below).
2. Choose email + display name + role, and a **temporary password** (min. 8 chars,
   e.g. `openssl rand -base64 12`).
3. Transmit this temporary password **only once, over a secure channel**: in person, or
   institutional messaging. **Avoid cleartext email.**
4. Ask the user to **change it on first login**
   (account menu → change password). The admin then forgets the temporary password.

> ⚠️ The email must use a real domain: `email-validator` **rejects** `.local`, `.test`, etc.

### Via the API (equivalent)

```bash
# 1. admin login -> cookie stored in admin.jar
curl -s -c admin.jar -X POST http://<BACKEND>/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"<ADMIN_PASSWORD>"}'

# 2. create the account (initial_password = temporary)
curl -s -b admin.jar -X POST http://<BACKEND>/api/v1/admin/users \
  -H 'Content-Type: application/json' \
  -d '{"email":"newuser@example.com","display_name":"First Last","role":"user","initial_password":"<TEMP>"}'
```

The `scripts/verify_compartmentalization.py` script automates creation + verification
of partitioning (useful for a test account).

---

## Reset a password (forgotten)

The admin **regenerates a temporary password** — they cannot recover the old one (hashed).

- Interface: Users page → reset.
- API: `PATCH /api/v1/admin/users/{user_id}` with `{"new_password":"<TEMP>"}`.

Side effect: `token_version` is incremented → all of this user's open sessions are
**immediately invalidated**. Same for deactivation (`is_active=false`)
and role change.

---

## Deactivate / revoke an account

`PATCH /api/v1/admin/users/{user_id}` with `{"is_active": false}`.
The current session is cut off on the next request (reload of `is_active`).
You **deactivate**, you do not delete (this preserves the `owner_id` attribution of the data).

---

## Infrastructure secrets → vault

Unlike user passwords, these secrets **must be retained**
(in a team password manager: Bitwarden/Vaultwarden, KeePassXC,
or the IT department (DSI) vault). Never commit them.

| Secret | Where it lives | Usage |
|--------|-----------|-------|
| `JWT_SECRET` | `deploy/dib-backend/.env` (gitignored, on dib) | JWT signing |
| `POSTGRES_PASSWORD` | same | Internal database `junon_db` |
| `BRGM_DB_PASSWORD` | same | Read access to the BRGM warehouse (gold) |
| Bootstrap admin account | team vault | Emergency admin access |

- The `.env` on dib remains the execution point, but a **reference copy** of the
  secrets must be in the vault (otherwise loss of the server = loss of the secrets).
- `COOKIE_SECURE=false` on dib (HTTP access). In TLS production (DSI K8s) → `true`.
- **K8s migration (DSI frontend)**: the cluster-side secrets will go into **K8s Secrets**.
  The `JWT_SECRET` stays on dib (backend), not affected by the cluster.

---

## Emergency admin account

Initial bootstrap via `scripts/create_admin.py`. Keep at least **one** active admin
account and its credentials in the vault. If the last admin is lost, recreate an
admin with this script directly on dib.
