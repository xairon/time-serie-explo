#!/usr/bin/env python3
"""Crée un compte de test (role=user) et vérifie le cloisonnement.

Usage: python3 scripts/verify_compartmentalization.py <BASE_URL> <ADMIN_EMAIL> <ADMIN_PASSWORD> <TEST_PASSWORD>
"""
import sys
import json
import urllib.request
import urllib.error
import http.cookiejar

BASE, ADMIN_EMAIL, ADMIN_PW, TEST_PW = sys.argv[1:5]
TEST_EMAIL = "test@example.com"


def client():
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar)), jar


def call(opener, method, path, body=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with opener.open(req, timeout=30) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = raw
        return e.code, payload


def main():
    # --- 1. login admin ---
    admin, _ = client()
    st, body = call(admin, "POST", "/api/v1/auth/login",
                    {"email": ADMIN_EMAIL, "password": ADMIN_PW})
    print(f"[admin] login -> {st}")
    assert st == 200, body

    # --- 2. create test user (idempotent) ---
    st, body = call(admin, "POST", "/api/v1/admin/users",
                    {"email": TEST_EMAIL, "display_name": "Compte Test",
                     "initial_password": TEST_PW, "role": "user"})
    if st == 409:
        print(f"[admin] create test user -> 409 (existe déjà), on récupère son id")
        st2, users = call(admin, "GET", "/api/v1/admin/users")
        test_id = next((u["id"] for u in users if u["email"] == TEST_EMAIL), None)
    else:
        print(f"[admin] create test user -> {st}")
        assert st in (200, 201), body
        test_id = body["id"]
    print(f"[admin] test user id = {test_id}")

    # --- 3. admin's models/datasets (for the 404 check) ---
    st, admin_models = call(admin, "GET", "/api/v1/models")
    admin_model_id = None
    if st == 200 and admin_models:
        # shape may be list or {items:[...]}
        items = admin_models.get("items") if isinstance(admin_models, dict) else admin_models
        if items:
            admin_model_id = items[0].get("run_id") or items[0].get("id") or items[0].get("model_id")
    print(f"[admin] models visible = {len(admin_models) if isinstance(admin_models, list) else admin_models}, sample id = {admin_model_id}")

    # --- 4. login test user ---
    test, _ = client()
    st, body = call(test, "POST", "/api/v1/auth/login",
                    {"email": TEST_EMAIL, "password": TEST_PW})
    print(f"[test]  login -> {st}")
    assert st == 200, body

    # --- 5. compartmentalization checks ---
    st, datasets = call(test, "GET", "/api/v1/datasets")
    n_ds = len(datasets) if isinstance(datasets, list) else (len(datasets.get("items", [])) if isinstance(datasets, dict) else "?")
    print(f"[test]  GET /datasets -> {st}, count = {n_ds} (attendu 0)")

    st, models = call(test, "GET", "/api/v1/models")
    n_m = len(models) if isinstance(models, list) else (len(models.get("items", [])) if isinstance(models, dict) else "?")
    print(f"[test]  GET /models -> {st}, count = {n_m} (attendu 0)")

    if admin_model_id:
        st, _ = call(test, "GET", f"/api/v1/models/{admin_model_id}")
        print(f"[test]  GET admin's model {admin_model_id[:12]} -> {st} (attendu 404)")

    st, _ = call(test, "GET", "/api/v1/admin/users")
    print(f"[test]  GET /admin/users -> {st} (attendu 403)")

    print("\n=== RESUME COMPTES ===")
    print(f"admin: {ADMIN_EMAIL} / {ADMIN_PW}")
    print(f"test : {TEST_EMAIL} / {TEST_PW}  (id={test_id})")


if __name__ == "__main__":
    main()
