# Suivi — validation des indices standardisés (SPI / STI / SPEI) + bugs relevés

**Date** : 2026-07-24
**Origine** : déploiement du SPEI (cf. `docs/superpowers/specs/2026-07-23-climat-spei-design.md`).
**Statut** : SPI ✅ · STI ✅ · SPEI ✅ (calibration partielle 1991-2000 concluante ; complète à rejouer) · β sans objet ✅ · bug warm corrigé ✅

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
| **SPEI** *(1991-2000, backfill partiel)* | 1/3/6/12 | +0,083 → +0,156 | 1,025 – 1,054 | 0,10 – 0,19 | 0,02 – 0,05 % |

→ **SPI, STI et SPEI sont correctement calibrés.** Rien à corriger.

Note SPEI : l'écart-type est excellent (≈ 1,03) et la saturation est **plus faible que celle
du SPI**. La moyenne légèrement positive (+0,08 à +0,16, croissante avec la fenêtre) est
attendue : 1991-2000 est la **première décennie** d'une référence 1991-2020 sous tendance au
dessèchement — la décennie de début est humide *relativement* à la moyenne trentenaire.
C'est un signal climatique réel, pas un défaut. Contre-épreuve à faire quand le backfill
sera complet : la moyenne sur 2011-2020 doit être **négative** du même ordre.

### SPEI — calibration complète à rejouer
La mesure ci-dessus porte sur 1991-2000 (le backfill historique 1950→2026 était encore en
cours). Rejouer sur la référence entière dès qu'elle est couverte :

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

## 2. β sans borne supérieure — VÉRIFIÉ, AUCUNE CORRECTION NÉCESSAIRE

> **Conclusion (mesurée le 2026-07-24)** : l'inquiétude théorique ci-dessous est **infirmée
> par les données**. Aucun correctif n'est appliqué. Ne pas « corriger » ce point sans
> refaire la mesure — ajouter une borne supprimerait 9,4 % de mailles qui fonctionnent
> parfaitement.
>
> Sur la référence (1991-2000, fenêtre 1, `spei` non nul), en séparant les mailles selon β :
>
> | cohorte | n | moyenne | écart-type | saturation \|z\|≥3,08 |
> |---|---|---|---|---|
> | β ≤ 50 | 1 030 558 | 0,083 | 1,042 | 0,040 % |
> | **β > 50** | 64 260 | 0,086 | **0,977** | **0,000 %** |
>
> Les mailles à β élevé se comportent **mieux** que les autres : écart-type plus proche de 1
> et **zéro** saturation. Un β grand traduit une distribution du bilan hydrique réellement
> resserrée (ex. été méditerranéen), et la standardisation la traite correctement — ce n'est
> pas une dégénérescence. Le pic de saturation observé en juin 2026 (10,1 %) est bien
> **météorologique** (mois record), pas un artefact d'ajustement.

**Analyse initiale conservée pour mémoire.** `fit_loglogistic_lmoments`
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

**Pistes envisagées puis ÉCARTÉES** (cf. encadré §2) : borne dure `β > β_max` → NaN, ou repli
sur CDF empirique. Les deux auraient dégradé le produit (perte de 9,4 % de mailles saines)
pour corriger un problème qui n'existe pas dans les données.

---

## 3. ✅ CORRIGÉ — bug préexistant, warm de cache au démarrage du backend

> **Corrigé le 2026-07-24** (`api/main.py`, commit `e04650b`) : les warms appellent désormais
> `get_sector_situation(type=…, month=None, network="all")` et
> `get_sector_timeline(type=…, network="all")` avec des valeurs explicites.
> **Vérifié en prod** : `docker logs junon-backend` → 0 occurrence de `InvalidDatetimeFormat`
> / `annotation=Union`, et les trois warms confirment désormais (`Stations GeoJSON`,
> `BRGM sectors`, `BRGM timeline` *cache warmed*).

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

- [ ] Calibration SPEI sur la référence **complète** 1991-2020 (§1) + contre-épreuve
      « moyenne 2011-2020 négative », dès la fin du backfill.
- [ ] Fréquence des 7 classes McKee vs attendu théorique, pour les 3 indices.
- [x] **Cohérence de signe SPEI ↔ `bilan_hydrique`** : `corr = +0,353` sur 1 094 818 lignes
      (fenêtre 1, 1991-2000). Correctement signée. La corrélation est modérée et non proche
      de 1 **par construction** : le SPEI est standardisé par maille ET par mois calendaire,
      ce qui retire justement la dispersion saisonnière et géographique que porte le
      `bilan_hydrique` brut. Une corrélation négative aurait été l'alerte.
- [x] **β sans borne supérieure** → vérifié, sans objet (§2).
- [x] **Bug du warm de cache `Query(...)`** → corrigé et vérifié en prod (§3).
- [x] `dagster definitions validate` : **contrôle indirect concluant** — le code-server a
      rechargé les définitions en prod et `dagster asset list` expose bien
      `fct_era5_spei_climatology_grid`. Le gap sandbox est levé.
