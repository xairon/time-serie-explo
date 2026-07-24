# Suivi — validation des indices standardisés (SPI / STI / SPEI) + bugs relevés

**Date** : 2026-07-24
**Origine** : déploiement du SPEI (cf. `docs/superpowers/specs/2026-07-23-climat-spei-design.md`).
**Statut** : SPI ✅ · STI ✅ · SPEI ✅ · couverture ✅ (100 %) · ETP ✅ **corrigée** (PEV ERA5 → Hargreaves FAO-56, 2,15× trop élevée) · bug warm ✅

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
| **SPEI** *(GLO + ETP Hargreaves)* | 1/3/6/12 | **+0,004 → +0,006** | **0,999 – 1,013** | 0,01 – 0,03 | **0,012 – 0,035 %** |

→ **SPI, STI et SPEI sont correctement calibrés.** Rien à corriger.

**SPEI — mesure définitive après bascule GLO** (référence complète 1991-2020, backfill de
41 960 400 lignes en 50,3 min) : moyenne nulle à la 3ᵉ décimale, écart-type ≈ 1,03-1,07,
saturation ≈ 0,045 % — soit **en dessous du SPI**. Le critère d'acceptation fixé *avant* la
mesure est atteint.

Point clé : le `n` est passé de 3,65 / 3,02 / 2,91 / 2,78 M (log-logistique, couverture
inégale selon la fenêtre) à **~4,14 M sur les quatre fenêtres** — la couverture est
désormais uniforme. La calibration est donc non seulement préservée mais mesurée sur
l'intégralité du domaine.

**SPI et STI vérifiés inchangés** après le rebuild (mêmes moyennes/écarts-types au millième
qu'avant la bascule) : le backfill ne les a pas perturbés.

**Contre-épreuve de la tendance — CONFIRMÉE.** La moyenne par décennie (fenêtre 1) décroît
de façon monotone :

| décennie | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|
| moyenne SPEI | **+0,038** | +0,007 | **−0,016** | **−0,117** |

La moyenne positive de la 1ʳᵉ décennie était donc bien un **signal climatique de
dessèchement**, pas un biais d'ajustement : prédiction faite avant mesure, vérifiée.

### Requête de contrôle (à rejouer si les marts amont changent)

```sql
SELECT fenetre, avg(spei), stddev_samp(spei),
       percentile_cont(0.5) WITHIN GROUP (ORDER BY spei),
       100.0*count(*) FILTER (WHERE abs(spei)>=3.08)/count(spei) AS sat_pct
FROM gold.fct_era5_indices_grid
WHERE month>='1991-01-01' AND month<'2021-01-01' AND spei IS NOT NULL
GROUP BY fenetre ORDER BY fenetre;
```

**Critère d'acceptation** (fixé *avant* la mesure) : moyenne ≈ 0 (±0,05), écart-type ≈ 1
(±0,05), saturation du même ordre que le SPI (< 1 %). → **Atteint** : moyenne ≤ 0,003 en
valeur absolue, écarts-types **0,999-1,013** (tous dans la tolérance), saturation ≤ 0,035 %.
La sur-dispersion résiduelle constatée avant la bascule ETP (w12 à 1,069) a **disparu**.

Réserve subsistante : la classe NORMAL représente 54,8 % au lieu des 59,9 % théoriques alors
que l'écart-type vaut 1,008 — la distribution garde des « épaules » un peu plus lourdes
qu'une gaussienne. Sans conséquence opérationnelle (les seuils McKee restent ceux du SPI et
du STI), mais à ne pas interpréter comme une anomalie de sécheresse : c'est une propriété de
la transformation GLO→normale près des bornes de classes.

**Contrôle de cohérence déjà fait (mois récents)** : la calibration suit correctement les
entrées brutes — mai 2026 médiane −0,59 (bilan −132,8 vs réf −105,3), avril −1,41
(−152,0 vs −78,7), juin −2,10 (−242,6, hors de toute la plage 1991-2020 : précip 49 mm
+ ETP 292 mm, deux records). Le pic de juin est donc **météorologique, pas artefactuel**.

---

## 2. β sans borne supérieure — SANS OBJET (paramètre disparu avec la bascule GLO)

> **Doublement clos.** (a) L'inquiétude était déjà **infirmée par les données** (encadré
> ci-dessous) ; (b) depuis la bascule vers la logistique généralisée (§3bis), le paramètre
> `β` **n'existe plus** — il est remplacé par `k = −τ₃`, borné par construction à |k| < 1.
> La question ne peut donc plus se poser. Section conservée pour la traçabilité du
> raisonnement.
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

## 3bis. ✅ RÉSOLU — couverture SPEI incomplète (cause trouvée, loi corrigée)

> **Résolu le 2026-07-24.** Cause identifiée puis corrigée : la log-logistique ne peut pas
> représenter les bilans hydriques à **asymétrie négative**. Remplacée par la **logistique
> généralisée (GLO)**. Détail complet et démonstration dans
> `docs/superpowers/specs/2026-07-23-climat-spei-design.md` §2.0.
>
> **Instrumentation** (ajoutée pour trancher, conservée) : `fit_reference_frame` compte
> désormais les rejets **par motif** et les journalise par fenêtre — plus de troncature
> silencieuse.
>
> | | avant (log-logistique) | après (GLO) |
> |---|---|---|
> | ajustement réussi | 74,6 % | **100,0 %** (zéro rejet) |
> | `spei` en carte (juin 2026) | 75 % | **99,2 %** |
> | motifs de rejet | 100 % `beta_hors_domaine` | — |
>
> **Non-régression prouvée** : sur les 35 614 mailles déjà ajustées avant, écart max
> **0,000** et corrélation **1,0000** entre ancienne et nouvelle valeur. Changement
> purement additif.

**Analyse initiale conservée pour mémoire.** Sévérité alors estimée moyenne-haute — c'était
bien le point le plus important de cet audit, plus que la piste β (§2) qui s'est révélée sans
objet.

Le SPEI n'est calculé que sur une fraction des mailles, contre 100 % pour le SPI :

| fenêtre | couverture SPEI | couverture SPI |
|---|---|---|
| 1 | 88,1 % | 100 % |
| 3 | 72,9 % | 100 % |
| 6 | 70,3 % | 100 % |
| 12 | **67,1 %** | 100 % |

**Ces trous ne sont pas dispersés, ils sont groupés** (couverture fenêtre 3, juin 2026, par
bande de latitude) :

| bande | 40-42 | 42-44 | **44-46** | 46-48 | 48-50 | **50-52** |
|---|---|---|---|---|---|---|
| couvert | 98,4 % | 92,0 % | **52,5 %** | 93,6 % | 81,2 % | **41,6 %** |

→ **Conséquence produit directe : la carte SPEI présente des trous régionaux visibles**
(bandes où près de la moitié des mailles sont vides), là où la carte SPI est pleine.
L'utilisateur verra une carte « à trous » sans explication.

**Origine** : chaque maille absente vient d'un groupe rejeté à l'ajustement dans
`fit_reference_frame` — soit `n < MIN_YEARS_REF` (25 ans), soit un ajustement non fini, soit
le garde `β ≤ 1`. La couverture du mart de référence (411 816 lignes sur 551 808 possibles,
soit 74,6 %) explique **exactement** la couverture observée du `spei`.

**À faire (dans cet ordre)** :
1. **Instrumenter** `fit_reference_frame` pour compter les rejets **par motif**
   (`n<25` / non fini / `β≤1`) et par fenêtre — aujourd'hui on ne peut pas distinguer les
   causes a posteriori, puisqu'un groupe rejeté ne laisse aucune ligne.
2. Selon le motif dominant : si `β ≤ 1` domine, envisager une paramétrisation plus robuste
   (log-logistique à 2 paramètres, ou Pearson III comme le fait `climate-indices`) **pour les
   seules mailles concernées**, plutôt que de les laisser vides.
3. En attendant, **assumer le trou dans l'UI** : la légende / le ⓘ doivent dire qu'une maille
   vide = ajustement impossible sur la référence, et non « pas de sécheresse ».

## 4. ✅ RÉSOLU — l'ETP n'était pas une ET0 de référence (bascule vers Hargreaves)

**Déclencheur** : comparaison avec l'étude World Weather Attribution sur l'Europe, qui
calcule son SPEI avec « ERA5 + schéma de **Hargreaves** » pour l'ETP. Nous utilisions la
`potential_evaporation` **native d'ERA5-Land**. Ce ne sont pas la même grandeur.

**Mesure** (30 888 mailles-mois, 2015-2025, mêmes mailles) :

| | Hargreaves | PEV ERA5-Land | ratio |
|---|---|---|---|
| ETP annuelle | **818 mm** | **1 756 mm** | **×2,15** |
| Bilan P−ETP | **+146 mm/an** | **−793 mm/an** | — |

818 mm/an est cohérent avec l'ET0 de référence pour la France (littérature 700-900 mm) ;
1 756 mm/an ne l'est pas et mettait le pays en déficit hydrique **permanent**. La PEV d'ERA5
n'est pas une ET0 FAO : c'est l'évaporation d'une surface sans stress hydrique calculée avec
la résistance aérodynamique du modèle.

**Correctif appliqué le 2026-07-24** : `etp_totale` est désormais une ET0 **Hargreaves
(FAO-56)** calculée depuis les Tmin/Tmax journaliers vrais (rendu possible par le cutover
température). La PEV brute est conservée en `etp_pev_era5`, non consommée.

**Effets mesurés — tous favorables :**
- ETP France **819 mm/an**, bilan **+150 mm/an** (le SQL reproduit un calcul Python
  indépendant à 1 mm près : 819 vs 818).
- Calibration SPEI **améliorée** : écart-type 1,032-1,069 → **0,999-1,013** ; saturation
  0,045 % → **0,024 %** ; couverture 99,9 % → **100 %**. La sur-dispersion résiduelle a
  disparu — signal indépendant que Hargreaves est la bonne ETP.
- Tendance décennale **moins amplifiée** : `+0,079 → −0,301` devient `+0,038 → −0,117`. Le
  dessèchement reste réel et monotone, mais la PEV l'exagérait d'un facteur ~2,5 (elle
  réagit plus fortement au réchauffement).
- Juin 2026 (w3) : médiane −2,16 → **−1,94**. Toujours sévère, moins extrême.

**Incohérence assumée** : la chaîne *station* (`hubeau_daily_chroniques`,
`hydro_daily_chroniques`, `fct_monthly_*`) garde la **PEV brute** — elle force les modèles
Pastas et la changer invaliderait tous les calages TFN. Une « ETP » du module Climat
(~820 mm/an) et une « ETP » de la page Station (~1 740 mm/an) ne sont donc pas la même
grandeur. À traiter comme un chantier distinct si on veut l'unifier.
`/observatory/era5/*` suit Hargreaves (il dérive sa PEV journalière de `etp_totale`).

---

## 5. Reste à vérifier

- [x] **Couverture SPEI (§3bis)** : instrumentée, cause trouvée (asymétrie négative), loi remplacée par la GLO → 100 %.

- [x] Calibration SPEI sur la référence **complète** 1991-2020 (§1) + contre-épreuve
      « moyenne 2011-2020 négative » : **les deux confirmés** (§1).
- [x] **Fréquence des 7 classes McKee vs théorie** (fenêtre 3, référence 1991-2020, en %) :

      | | ExtBas | TrèsBas | Bas | NORMAL | Haut | TrèsHaut | ExtHaut |
      |---|---|---|---|---|---|---|---|
      | théorie N(0,1) | 4,0 | 6,0 | 10,0 | **59,9** | 10,0 | 6,0 | 4,0 |
      | SPI | 4,6 | 6,2 | 9,5 | 58,9 | 11,1 | 6,4 | 3,4 |
      | STI | 3,1 | 6,8 | 10,8 | 58,9 | 9,7 | 6,7 | 4,1 |
      | **SPEI** (GLO+Hargreaves) | 3,8 | **7,7** | 10,9 | **54,8** | 11,6 | **7,7** | 3,5 |

      SPI et STI collent à la théorie. Le **SPEI est légèrement sur-dispersé** : classe
      NORMAL à 53,5 % au lieu de 59,9 %, au profit des classes « très sec » / « très
      humide » (+1,5 et +2,3 pts). C'est **cohérent avec son écart-type mesuré (1,038)** :
      une distribution un peu plus large peuple davantage les queues. Conséquence produit à
      connaître : **le SPEI signalera « très sec » un peu plus souvent que le SPI**, ce qui
      est en partie l'effet recherché (il intègre l'ETP) mais en partie un artefact de
      dispersion. Non bloquant ; à re-mesurer si l'ajustement change (§3bis).
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
