# Effet du forçage d'évapotranspiration sur les modèles métier

**Mesuré le 25 août 2026.** Cette note consigne une vérification ponctuelle, afin qu'elle soit
rejouable et qu'elle ne repose pas sur un souvenir.

## Question

La chaîne des stations expose `potential_evaporation`, la variable brute d'ERA5-Land, et non
l'évapotranspiration de référence calculée par Hargreaves dans la chaîne climatique sur grille
(voir `hubeau_data_integration/docs/ERA5.md`). Les modèles Pastas sont donc calibrés sur un
forçage plus élevé que l'ET0 de référence. Le paramètre de pondération de l'évaporation
absorbe-t-il cet écart, laissant les prévisions inchangées ?

## Mesure du forçage

```sql
SELECT round(avg(-potential_evaporation)::numeric, 3)         AS mm_par_jour,
       round((avg(-potential_evaporation) * 365)::numeric, 0) AS mm_par_an
FROM gold.hubeau_daily_chroniques;
```

Résultat : **5,039 mm/jour, soit 1 839 mm/an**, contre environ 818 mm/an pour l'ET0 de référence.

## Protocole

Un même modèle est calibré deux fois sur la même station, avec le même modèle de recharge
(`Linear`) et la même réponse (`Gamma`), en ne changeant que l'échelle de l'évaporation :
d'abord la variable brute, puis cette même variable divisée par 2,25 pour la ramener à l'ordre
de grandeur de l'ET0 de référence.

Station : `08077X0030/ERH`. Profondeur disponible : 385 jours (la base ne portait alors qu'un
chargement partiel, du 2025-01-01 au 2026-08-23).

## Résultats

| | Variable brute | Échelle corrigée |
|---|---|---|
| Évaporation moyenne | 5,857 mm/j (2 138 mm/an) | 2,603 mm/j (950 mm/an) |
| Variance expliquée | 72,16 % | 68,54 % |
| Nash-Sutcliffe | 0,7216 | 0,6854 |
| RMSE | 0,2770 | 0,2944 |
| Paramètre `rch_f` | **−2,0000 (borne)** | **−2,0000 (borne)** |

## Ce que la mesure établit

Le paramètre de pondération **n'absorbe pas** l'écart d'échelle : il sature à sa borne dans les
deux cas. L'ajustement dépend donc réellement du forçage employé.

Le meilleur ajustement obtenu avec la variable brute ne signifie pas qu'elle soit la bonne. Un
paramètre collé à sa borne traduit un modèle qui réclame plus d'évaporation que son forçage ne
lui en fournit, ce qui est le symptôme attendu d'une entrée mal dimensionnée.

## Portée

Une seule station, 385 jours, un seul couple recharge/réponse. La mesure établit que l'écart a
un effet, pas son ampleur générale. Une évaluation sur un historique long et un échantillon de
stations reste à conduire.

## Suite

Corriger suppose d'ajouter les températures extrêmes journalières à `int_era5_for_all_stations`,
d'y calculer l'ET0 de Hargreaves, de propager dans `hubeau_daily_chroniques` et
`hydro_daily_chroniques` (deux hypertables, donc reconstruction complète), puis de recalibrer les
modèles existants.
