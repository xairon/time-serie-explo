# Production hardening: security + RGPD + Streamlit decommission

Date: 2026-06-05
Status: Approved (user: "fait au mieux" — recommended options adopted)

## Goal

Make the application safe to expose publicly and legally acceptable for accounts
holding personal data (emails), and remove the legacy Streamlit UI to cut
maintenance surface. Builds on the already-merged `fix(security)` commit
(fail-closed secrets + task ownership).

## Decisions (locked)

- **RGPD scope:** Minimal — right to erasure, retention/purge, privacy notice.
- **Streamlit:** remove Streamlit UI + dependency, keep `dashboard/utils/`
  (shared business library imported by the API).
- **Security items:** rate limiting + lockout, admin-initiated password reset,
  headers/CORS hardening, auth audit log.
- **Cleanup:** conservative (only clearly-dead code, verified by usage search +
  green tests). Do not delete `*-dev/` runtime data directories.
- **Account deletion:** full erasure — delete the account AND resources it owns
  (datasets, tasks, MLflow runs tagged `owner_id`). Orphan artifacts are already
  denied to non-admins.
- **Split:** two PRs. PR1 additive (security + RGPD). PR2 destructive (Streamlit
  removal + cleanup).
- **Lockout storage:** Redis with TTL (no schema churn, auto-expiry).

## Context discovered

- `User` model (`api/models_db/user.py`): has `id, email, display_name,
  password_hash, role, is_active, token_version, created_at, last_login_at`.
  No `must_change_password`, no failed-login columns.
- `dashboard/utils/` does NOT import Streamlit at module level (the `streamlit`
  string matches were identifiers/comments e.g. `StreamlitSafeUnpickler` in
  `robust_loader.py`, which IS used by `model_registry.py`). Removing the
  `streamlit` dependency is therefore safe for the API.
- Streamlit UI footprint: `dashboard/training/` (Home.py + 4 pages) and
  `dashboard/components/` (Streamlit UI components).
- nginx already sets X-Content-Type-Options, X-Frame-Options, Referrer-Policy,
  CSP. Missing: HSTS, Permissions-Policy.
- No SMTP/email infra → password reset is admin-initiated (no self-service
  forgot-password flow).
- Redis is available (`settings.redis_url`, `api/cache.py`).
- Only 1 Alembic migration exists.

---

## PR 1 — Security + RGPD (additive)

### A. Rate limiting + account lockout

- Add `slowapi` (Redis storage via existing `redis_url`, in-memory fallback).
- IP rate limit on `POST /auth/login` (~5/min/IP) + a permissive global default.
- Per-account lockout in Redis: key `login_fail:{email_lower}` incremented on
  failure with a sliding TTL; at 5 failures within window → lock 15 min
  (`login_lock:{email_lower}` with TTL). Reset counter on success.
- Responses stay generic (401 for bad creds, 429 when rate-limited/locked) — no
  user enumeration. Lockout is keyed on submitted email regardless of existence.

### B. Auth audit log

- New table `auth_events`: `id (uuid pk)`, `user_id (uuid, nullable, no FK
  cascade dependency — keep event after user deletion)`, `email (string)`,
  `event_type (string)`, `ip (string, nullable)`, `user_agent (string,
  nullable)`, `created_at (timestamptz default now)`.
- Event types: `login_success`, `login_failure`, `logout`, `password_change`,
  `password_reset`, `account_deleted`, `admin_user_created`,
  `admin_user_updated`, `admin_user_deleted`.
- Write from `api/routers/auth.py` and `api/routers/admin.py` via a small helper
  `api/auth/audit.py::record_event(db, ...)`.
- Admin read endpoint `GET /api/v1/admin/auth-events` (paginated, newest first).
- Alembic migration for the table.

### C. Headers / CORS hardening

- `nginx/nginx.conf` and `deploy/frontend/nginx.conf.template`: add
  `Strict-Transport-Security "max-age=31536000; includeSubDomains" always;` and
  `Permissions-Policy "geolocation=(), microphone=(), camera=()" always;`.
- Keep CSP as-is (the `unsafe-inline`/`unsafe-eval` are required by the Vite
  build; tightening is logged as future debt, not in scope).
- API CORS already minimal (no `*`, no credentials); leave as-is.

### D. Password reset (admin-initiated)

- Migration: add `User.must_change_password: bool default false not null`.
- `POST /api/v1/admin/users/{id}/reset-password`: generate a strong random
  temporary password, set `password_hash`, bump `token_version` (revoke
  sessions), set `must_change_password=True`, write `password_reset` audit
  event, return `{ "temporary_password": "..." }` ONCE (admin transmits
  out-of-band).
- `GET /auth/me` exposes `must_change_password`. Frontend forces a change via the
  existing `/auth/change-password`, which clears the flag.
- Self-service forgot-password is explicitly out of scope (needs SMTP);
  documented as future work.

### E. RGPD minimal

- **Erasure:** `DELETE /api/v1/auth/me` (self) and `DELETE
  /api/v1/admin/users/{id}` (admin). Deletes:
  - the `users` row;
  - datasets owned by the user (their `prepared/<id>/` dirs via
    `DatasetRegistry`);
  - in-memory tasks owned by the user (`task_manager`);
  - MLflow runs tagged `owner_id = <user>` (best-effort delete via existing
    registry helpers).
  - Writes an `account_deleted` audit event (kept; `user_id` retained for
    legal traceability of the deletion itself, email may be redacted).
  - Admins cannot delete the last remaining admin (guard).
- **Retention/purge:** `scripts/purge_expired.py` — deletes `auth_events` older
  than `AUTH_EVENT_RETENTION_DAYS` (default 365) and is safe to run repeatedly;
  documented for cron. Completed tasks >24h are already purged by `task_manager`.
- **Privacy notice:** frontend route `/privacy` (public), linked from the login
  page. Structured content: data processed (email, display name, uploaded hydro
  data), purpose, retention, user rights (access/rectification/erasure),
  contact. Marked "à valider par le DPO BRGM". A short `PRIVACY.md` mirrors it.

### Config additions

- `AUTH_EVENT_RETENTION_DAYS: int = 365`
- `LOGIN_RATE_LIMIT: str = "5/minute"`
- `LOGIN_MAX_FAILURES: int = 5`, `LOGIN_LOCKOUT_MINUTES: int = 15`

### Dependencies

- Add `slowapi` to the `api` extra in `pyproject.toml` / requirements.

### Tests (PR1)

- Rate limit returns 429 after threshold.
- Lockout: N failures → locked → correct creds still 429 until TTL.
- Password reset: admin gets temp pwd, old sessions invalid, `must_change`
  enforced, change clears flag.
- Account deletion: user gone, owned dataset dir removed, sessions invalid,
  audit event present; last-admin guard.
- Audit events written for login success/failure.

---

## PR 2 — Streamlit removal + conservative cleanup (destructive)

- Delete `dashboard/training/` and `dashboard/components/` (Streamlit UI).
- For each `dashboard/utils/*` flagged by a `streamlit` text match
  (`robust_loader.py`, `timeshap_wrapper.py`, `custom_wrappers.py`): confirm no
  top-level `import streamlit`; keep if imported by `api/` or retained utils
  (e.g. `robust_loader` is used by `model_registry`), otherwise remove if
  unreferenced.
- Remove `streamlit` and Streamlit-only deps from `requirements/base.txt`,
  `requirements.txt`, `pyproject.toml`.
- Update `ARCHITECTURE.md`, `DEPLOYMENT.md`, `README.md` to drop Streamlit.
- Verify `*-dev/` dirs are gitignored; do NOT delete their contents.

### Guardrails (PR2)

- `rg -n streamlit` (excluding `.venv`, `node_modules`, docs history) returns
  nothing in retained code.
- `pytest tests/` green (tests import `dashboard.utils`).
- `cd frontend && npm run build` green.

---

## Out of scope (logged as future work)

- Self-service forgot-password (needs SMTP).
- Multi-replica backend: in-memory `task_manager` → Redis-backed job queue.
- Observability wiring (prometheus/otel deps present but unused).
- CSP tightening to drop `unsafe-inline`/`unsafe-eval`.
- CI on `dev`/PRs and a lint gate.
