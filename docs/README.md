# Documentation map

Every document in this repository, by what you are trying to do. Nothing lives here that is
not listed below; if you add a document, add its line too.

**Verified on 2026-08-24.**

## Understand what this is

| Document | Read it when |
|----------|--------------|
| [../README.md](../README.md) | You are new. What Junon does, how to install it, and how to create the first account — without which you cannot log in. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | You need the shape of the system: the three layers, how training stays independent of any interface, the project tree. |
| [observatory.md](observatory.md) | You work on the Observatory or the Climat tab: views, map layers, API surface, cache purge. |

## Look something up

| Document | Read it when |
|----------|--------------|
| [climate-indices.md](climate-indices.md) | You need to know what SPI / STI / SPEI mean here, how they were validated, and the two results that are easy to misread. |
| [etp-station-mesure-2026-08-25.md](etp-station-mesure-2026-08-25.md) | 🗄 Snapshot du 25/08/2026. Mesure de l'effet du forçage d'évapotranspiration sur une calibration Pastas : le paramètre sature, l'écart n'est pas absorbé. |
| [account-management.md](account-management.md) | You are creating accounts, assigning roles, or handling secrets and password policy. |

## Do something

| Document | Read it when |
|----------|--------------|
| [DEPLOYMENT.md](DEPLOYMENT.md) | You are deploying. Production is split frontend/backend. |
| [../deploy/README.md](../deploy/README.md) | You need the split-deployment overview. |
| [../deploy/frontend/README.md](../deploy/frontend/README.md) | You are wiring the frontend onto the Kubernetes cluster. |
| [dev-environment.md](dev-environment.md) | You are setting up the isolated dev environment on dib. |

## Legal

| Document | Note |
|----------|------|
| [../PRIVACY.md](../PRIVACY.md) | Privacy notice, **in French on purpose**: it mirrors the in-app page shown to French users and is a legal notice, not developer documentation. |

## Related repository

The ERA5 grid marts, the standardized indices and the whole warehouse behind the Observatory
live in **`hubeau_data_integration`**. Anything about how an index is *computed* — the gamma /
GLO fits, the Hargreaves reference PET, the rebuild procedures — is documented there, in its
`docs/ERA5.md`. Junon reads those tables; it does not produce them.

Two pieces of maths are duplicated across the repositories on purpose and are guarded by
matching golden tables — see the cross-repository contract in
[climate-indices.md](climate-indices.md#cross-repository-contract).

## Test suite — read this before trusting it

**552 tests exist and nothing runs them automatically.** `.gitlab-ci.yml` has a single `build`
stage; the only test automation is `.github/workflows/test.yml`, a GitHub Actions workflow, and
this repository lives on GitLab. `tests/` is not copied into the backend image and `pytest`
sits in the `dev` extra that the image does not install.

Measured on 2026-08-24 — **547 passed, 4 failed, 1 skipped** in 92 s. None of the four failures
means the application is broken; they mean the suite mixes three kinds of test without saying so.

| Failing test | Error | What it actually is |
|--------------|-------|---------------------|
| `test_brgm_sectors_parse` | `FileNotFoundError: /tmp/bdrv/wfs_2026-05-01.json` | **A broken test.** It reads a hardcoded absolute path into someone's `/tmp`. It cannot pass on any other machine, ever. Either commit the snapshot as a fixture or delete the test. |
| `test_sectors_endpoint` | `relation "gold.dim_piezo_stations" does not exist` | **An integration test in disguise.** It needs a populated warehouse. It fails on a fresh install and passes once dbt has run. |
| `test_sectors_timeline_endpoint` | idem | idem |
| `test_purge_expired` | `TypeError: can't compare offset-naive and offset-aware datetimes` | **A harness artifact.** The suite runs on `sqlite+aiosqlite:///:memory:`, which does not preserve timezone awareness; production uses PostgreSQL with `DateTime(timezone=True)` and compares correctly. The code is right — but the GDPR retention purge is effectively untested. |

The honest reading: **548 of the 552 are real unit tests and they pass.** Two require a
warehouse, one requires a file nobody has, and one is defeated by the test database. Splitting
the integration tests into their own marker would make a green run mean something.

To run them against the real dependency set:

```bash
docker cp tests junon-backend:/app/tests
docker exec junon-backend pip install -q pytest pytest-asyncio aiosqlite
docker exec -w /app junon-backend python3 -m pytest tests -o addopts="" -q
```

## Conventions

- **One fact, one place.** A fact lives in exactly one document; everywhere else links to it.
- **Every document is listed here.** A document absent from this map is unfindable, which
  makes it worse than no document.
- **Dates are claims.** The "verified on" date says when someone last checked the content
  against the code. Update it when you check, not when you only edit prose.
- **Design notes, implementation plans and audit snapshots do not live in the repository.**
  46 such files were removed on 2026-08-24. What survived from them is folded into the
  documents above; the rest is in the Git history (`git log --diff-filter=D --name-only` to
  find a file, `git show <sha>^:<path>` to read it).
