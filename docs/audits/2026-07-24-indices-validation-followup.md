# Suivi — validation des indices standardisés (SPI / STI / SPEI) + bugs relevés

**Date** : 2026-07-24
**Origine** : déploiement du SPEI (cf. `docs/superpowers/specs/2026-07-23-climat-spei-design.md`).
**Statut** : SPI ✅ · STI ✅ · SPEI ⏳ (calibration en attente du backfill historique)

Objectif : s'assurer que les indices sont **calculés correctement** et **font sens**, et
recenser les bugs à corriger.

---

## 1. Calibration — le test de référence

Un indice standardisé doit suivre ~N(0,1) **sur sa propre période de référence
1991-2020**. Mesuré sur `gold.fct_era5_indices_grid` (41 960 400 lignes, 11 496 mailles) :

| Indice | fenêtres | moyenne | écart-type | médiane | saturation \|z\|≥3.08 |
|---|---|---|---|---|---|
| **SPI** | 1/3/6/12 | −0,008 → +0,002 | 0,985 – 1,031 | 0,00 – 0,04 | 0,05 – 0,51 % |
| **STI** | 1/3/6/12 | 0,000 | 0,983 | −0,09 – −0,02 | 0,00 – 0,04 % |
| **SPEI** | 1/3/6/12 | ⏳ | ⏳ | ⏳ | ⏳ |

→ **SPI et STI sont correctement calibrés.** Rien à corriger.

### SPEI — à compléter
Le backfill historique de `spei` (1950→2026) était encore en cours à la rédaction.
Rejouer la même requête dès qu'il couvre 1991-2020 :

```sql
SELECT fenetre, avg(spei), stddev_samp(spei),
       percentile_cont(0.5) WITHIN GROUP (ORDER BY spei),
       100.0*count(*) FILTER (WHERE abs(spei)>=3.08)/count(spei) AS sat_pct
FROM gold.fct_era5_indices_grid
WHERE month>='1991-01-01' AND month<'2021-01-01' AND spei IS NOT NULL
GROUP BY fenetre ORDER BY fenetre;
```

**Critère d'acceptation** : moyenne ≈ 0 (±0,05), écart-type ≈ 1 (±0,05), saturation du
même ordre que le SPI (< 1 %). Un écart net signalerait un défaut d'ajustement (§2).

**Contrôle de cohérence déjà fait (mois récents)** : la calibration suit correctement les
entrées brutes — mai 2026 médiane −0,59 (bilan −132,8 vs réf −105,3), avril −1,41
(−152,0 vs −78,7), juin −2,10 (−242,6, hors de toute la plage 1991-2020 : précip 49 mm
+ ETP 292 mm, deux records). Le pic de juin est donc **météorologique, pas artefactuel**.

---

## 2. À corriger — pas de borne supérieure sur β (log-logistique SPEI)

**Sévérité : moyenne.** `fit_loglogistic_lmoments`
(`hubeau_data_integration/src/hubeau_pipeline/ml/era5_indices.py`) rejette `β ≤ 1`
(divergence de Γ(1−1/β)) mais **n'impose aucune borne supérieure**.

Mesuré sur `gold.fct_era5_spei_climatology_grid` (411 816 lignes) :

| p50 | p90 | p99 | p99,9 | max |
|---|---|---|---|---|
| 11,0 | 47,3 | 443,2 | 3 975 | **787 081** |

→ **9,43 % des groupes ont β > 50**, 0,43 % ont β > 1000.

**Pourquoi c'est un problème** : `F(x) = [1 + (α/(x−γ))^β]⁻¹`. Quand β est grand, la CDF
tend vers une marche en `x = γ+α` : tout écart minime bascule F de ~0 à ~1, donc le SPEI
sature au clip ±3,09. À β = 47, la plage F ∈ [0,01 ; 0,99] tient dans ±10 % autour de la
médiane — l'indice perd toute résolution sur ces mailles.

**Cause probable** : ajustement par L-moments sur ~30 échantillons seulement ; les mailles
dont le bilan hydrique est très resserré (été méditerranéen : chaque juillet ≈ −120 ± 10 mm)
produisent un rapport de L-moments proche de la dégénérescence.

**Pistes de correction** (à trancher) :
- Borne dure : rejeter `β > β_max` (ex. 100) → NaN, comme les autres cas dégénérés.
- Ou repli sur une CDF empirique pour ces mailles (⚠ contraire au choix « pas de repli
  distributionnel » acté dans la spec §7 — à rediscuter si retenu).
- Dans tous les cas : **journaliser le taux de mailles rejetées par région** (pas de
  troncature silencieuse).

---

## 3. À corriger — bug préexistant, warm de cache au démarrage du backend

**Sévérité : basse (non bloquante), mais réelle.** Au démarrage de `junon-backend`, un warm
de cache passe l'**objet FastAPI `Query(...)` au lieu de sa valeur** comme paramètre `month` :

```
sqlalchemy.exc.DataError: (psycopg2.errors.InvalidDatetimeFormat)
invalid input syntax for type date:
"annotation=Union[str, NoneType] required=False default=None alias='month'
 json_schema_extra={} metadata=[_PydanticGeneralMetadata(pattern='^\d{4}-\d{2}$')]-01"
```

Requête concernée : `gold.fct_monthly_index` avec `type='hydro'`. Visible via
`docker logs junon-backend`. **Sans rapport avec le SPEI** (vérifié : `api/main.py` et les
chemins `fct_monthly_index` ne sont pas touchés par le merge `bfd9994..cb74b07`).

Cause typique : une fonction de route appelée **directement** (hors routeur), si bien que
le défaut `Query(...)` n'est jamais résolu. Correctif : passer une valeur explicite au warm.

Conséquence actuelle : le warm échoue, l'application démarre quand même (`healthy`), mais le
cache visé n'est pas préchauffé → première requête utilisateur plus lente.

---

## 4. Point de méthode assumé (pas un bug)

L'ETP provient de la **PEV ERA5-Land**, qui n'est **pas** un Penman-Monteith FAO-56. Le SPEI
en hérite. C'est cohérent avec l'ETP et le bilan hydrique déjà affichés, et le caveat est
surfacé dans l'UI (`climat.picker.speiInfo`, ⓘ sur SPEI, fr+en). Améliorer l'ETP est un
chantier data distinct.

---

## 5. Reste à vérifier

- [ ] Calibration SPEI 1991-2020 (§1) dès la fin du backfill.
- [ ] Fréquence des 7 classes McKee vs attendu théorique, pour les 3 indices.
- [ ] `dagster definitions validate` en CI/staging (jamais exécutable en sandbox ;
      **contrôle indirect déjà OK** : le code-server a rechargé les définitions en prod avec
      le nouvel asset visible dans `dagster asset list`).
- [ ] Cohérence de signe SPEI ↔ `bilan_hydrique` sur un échantillon de mailles.
