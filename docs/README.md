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
