# Pastas Realistic Scenarios — Design Spec

**Date**: 2026-04-24
**Approach**: Backend Authority, Frontend Cache (Approach 3)

## Problem

The current scenario system allows arbitrary values with no physical plausibility checks. Users can set 10,000 m³/d pumping on fractured bedrock or irrigation in January without any feedback. The 4 hardcoded presets (drought, constant pumping, climate trend, PET increase) use fixed values disconnected from the actual aquifer context.

## Goals

1. Realistic default values for pumping scenarios based on **usage type** (AEP, irrigation, industrial) and **aquifer family**
2. Double-layer validation: hard limits (reject absurd) + soft warnings (flag unusual)
3. Contextual preset scenarios that adapt to the selected station's aquifer
4. Save/reload named scenarios per model, with cross-model reuse

## Non-Goals

- Geothermal or temporary dewatering pumping types
- Spatial well placement on a map
- Multi-well interference calculation
- Real-time aquifer test data integration

---

## 1. Aquifer Family Classification

5 families derived from BDLISA codes already in `config.py`:

| Family | BDLISA codes (nature_eh / milieu_eh) | Typical productivity |
|--------|--------------------------------------|---------------------|
| `alluvial` | nature=3 | High (50-500 m³/h) |
| `sedimentary` | nature=5_1 | Medium-high (20-200 m³/h) |
| `karst` | nature=4, milieu=3 | Very variable (10-1000 m³/h) |
| `fractured` | nature=5_2, nature=0 (bedrock) | Low (1-30 m³/h) |
| `volcanic` | nature=6, nature=7 (mountain) | Variable (5-100 m³/h) |

**Auto-detection**: `detect_aquifer_family(station_metadata) -> AquiferFamily` reads `nature_eh`/`milieu_eh` from station metadata (available in BDLISA presets). Falls back to `sedimentary` with a warning if metadata is missing.

Mapping function lives in `scenario_presets.py` alongside the referential.

---

## 2. Pumping Usage Profiles

3 usage types, each with a characteristic temporal pattern:

### AEP (Drinking Water Supply)
- **Pattern**: Near-constant with summer peak (+20-30% in June-August)
- **Seasonality**: All 12 months active, peak months [6, 7, 8]
- **Typical duration**: Permanent (multi-year)

### Irrigation
- **Pattern**: Strongly seasonal
- **Seasonality**: Active months [4, 5, 6, 7, 8, 9], peak [6, 7, 8]
- **Typical duration**: Seasonal (April-September each year)

### Industrial
- **Pattern**: Constant year-round
- **Seasonality**: All 12 months, no peak
- **Typical duration**: Permanent

---

## 3. Referential: Usage x Aquifer Family

File: `dashboard/utils/pastas/scenario_presets.py`

Structure per combination:

```python
@dataclass
class PumpingRange:
    default: float        # Pre-filled value
    typical_min: float    # Soft warning below
    typical_max: float    # Soft warning above
    hard_max: float       # HTTP 422 above

@dataclass
class PumpingProfile:
    rate_m3d: PumpingRange
    distance_m: PumpingRange
    pattern: str                      # "constant" | "seasonal"
    active_months: list[int]          # [1..12]
    peak_months: list[int]            # Months with higher rate
    peak_factor: float                # Multiplier during peak (e.g., 1.25)
    typical_duration_days: int
    rfunc: str                        # Default response function
```

### Rate referential (m³/d)

| Usage | alluvial | sedimentary | karst | fractured | volcanic |
|-------|----------|-------------|-------|-----------|----------|
| **AEP** | 300 [100-800] max 5000 | 200 [50-500] max 3000 | 400 [50-1000] max 8000 | 30 [10-80] max 500 | 80 [20-200] max 1500 |
| **Irrigation** | 500 [100-1500] max 5000 | 300 [50-800] max 3000 | 600 [100-2000] max 8000 | 20 [5-50] max 300 | 60 [10-150] max 1000 |
| **Industrial** | 200 [50-500] max 3000 | 150 [30-400] max 2000 | 300 [50-800] max 5000 | 15 [5-40] max 200 | 50 [10-120] max 800 |

Format: `default [typical_min-typical_max] max hard_max`

### Distance referential (m)

Common across usages per aquifer family:

| Family | default | typical | hard_min | hard_max |
|--------|---------|---------|----------|----------|
| alluvial | 500 | [200-2000] | 10 | 50000 |
| sedimentary | 1000 | [300-5000] | 10 | 50000 |
| karst | 1000 | [100-5000] | 10 | 50000 |
| fractured | 300 | [100-1000] | 10 | 20000 |
| volcanic | 500 | [150-2000] | 10 | 30000 |

### Non-pumping hard limits

| Modification | Parameter | Hard limits |
|-------------|-----------|-------------|
| `scale_stress` | `factor` | [0.1, 5.0] |
| `linear_trend` | `slope_m_per_year` | [-1.0, 1.0] |
| All | `end - start` | >= 1 day |

---

## 4. Validation System

### 4.1 Hard validation (backend — reject)

Enforced in `validate_modifications(modifications, aquifer_family)`.

The `aquifer_family` is resolved server-side from the model's station metadata (MLflow tags on the run). The frontend never needs to send it explicitly for validation — it's derived from `run_id`.

- `rate_m3d` > `hard_max` for the aquifer family → HTTP 422
- `rate_m3d` < 0 → HTTP 422
- `distance_m` outside `[hard_min, hard_max]` → HTTP 422
- `factor` outside `[0.1, 5.0]` → HTTP 422
- `slope_m_per_year` outside `[-1.0, 1.0]` → HTTP 422
- `end` <= `start` → HTTP 422
- Sum of all pumping `rate_m3d` > 2× `hard_max` of the aquifer → HTTP 422

### 4.2 Soft validation (backend — warn)

Returns warnings in `ScenarioResponse.warnings[]` and also via pre-validation endpoint:

- `rate_m3d` outside `[typical_min, typical_max]` → "Débit de {X} m³/j inhabituel pour un pompage {usage} sur nappe {famille} — plage typique : {min}-{max} m³/j"
- `distance_m` < 50m → "Distance très faible ({X}m), l'impact piézométrique pourrait être surestimé"
- Irrigation pumping with active months outside [4-9] → "Pompage d'irrigation hors période végétative"
- `scale_stress` factor < 0.5 → "Réduction de {stress} de {pct}% — scénario très sévère"
- Combined severe: precip factor < 0.7 AND total pumping > typical_max → "Scénario combiné très contraint — sécheresse + pompage fort"
- Cross-model apply with different aquifer family → "Scénario calibré sur nappe {A}, appliqué sur nappe {B} — vérifiez les ordres de grandeur"

### 4.3 Frontend validation (instant, from cache)

- Input fields `min`/`max` attributes bound to hard limits
- Inline yellow warning below field when value is outside typical range
- Warnings computed locally from cached presets, no API call needed
- "Simuler" button always enabled if hard limits respected

### 4.4 Pre-validation endpoint

`POST /api/v1/pastas/validate-modifications`

```python
class ValidateRequest(BaseModel):
    modifications: list[Modification]
    aquifer_family: Optional[AquiferFamily] = None

class ValidateResponse(BaseModel):
    valid: bool                    # True if all hard limits pass
    errors: list[str]              # Hard limit violations
    warnings: list[str]            # Soft warnings
```

---

## 5. Contextual Preset Scenarios

Replace the 4 hardcoded presets with dynamic, aquifer-aware templates.

### 5.1 Preset definitions

Each preset is a template that generates `Modification[]` from the referential:

```python
@dataclass
class ScenarioPreset:
    id: str
    name: str
    description: str
    icon: str                              # Emoji for UI card
    build_modifications: Callable[[AquiferFamily, date, date], list[dict]]
```

**6 presets**:

| ID | Name | Modifications generated |
|----|------|------------------------|
| `aep_well` | Nouveau forage AEP | 1× pumping_synthetic (usage=aep, values from referential) |
| `irrigation` | Irrigation saisonnière | 1× pumping_synthetic (usage=irrigation, seasonal pattern) |
| `industrial` | Prélèvement industriel | 1× pumping_synthetic (usage=industrial, constant) |
| `summer_drought` | Sécheresse estivale | 1× scale_stress precip ×0.7 on Jun-Sep |
| `prolonged_drought` | Sécheresse prolongée | 1× scale_stress precip ×0.8 (2yr) + 1× scale_stress evap ×1.1 (2yr) |
| `climate_trend` | Tendance climatique | 1× linear_trend -0.02 m/yr + 1× scale_stress evap ×1.05 |

### 5.2 API endpoint

`GET /api/v1/pastas/scenario-presets?aquifer_family={family}&tmin={date}&tmax={date}`

Returns the full referential + pre-built presets with concrete values:

```python
class ScenarioPresetsResponse(BaseModel):
    aquifer_families: dict[str, str]                    # id → display name
    pumping_profiles: dict[str, dict[str, PumpingProfile]]  # usage → family → profile
    non_pumping_limits: dict                             # scale_stress, linear_trend limits
    presets: list[PresetScenario]                        # Pre-built with concrete modifications
```

Frontend calls this once on page mount, caches with React Query (stale 30min).

### 5.3 Frontend UX

Page layout (sidebar left):
1. Model picker (unchanged)
2. Simulation window (unchanged)
3. **"Scénarios prêts à l'emploi"** — grid of clickable cards (the 6 presets)
   - Each card shows: icon, name, 1-line description, aquifer badge
   - Click → fills modification list, user can adjust before simulating
4. **"Composer un scénario"** — current composer with additions:
   - When adding `pumping_synthetic`, first pick **usage** (AEP / irrigation / industriel / personnalisé)
   - Fields pre-filled from referential based on usage + detected aquifer
   - Soft warnings inline under each field
5. **"Mes scénarios"** — list of saved scenarios (see section 6)

---

## 6. Scenario Persistence

### 6.1 Storage

MLflow artifacts on the model's run, in `scenarios/` subfolder:

```
artifacts/
  scenarios/
    scenario_aep_2024.json
    scenario_secheresse.json
```

JSON schema:

```json
{
  "name": "Nouveau forage AEP Boissy",
  "description": "Forage AEP 300 m³/j à 500m, nappe alluviale",
  "created_at": "2026-04-24T14:30:00",
  "aquifer_family": "alluvial",
  "tmin": "2020-01-01",
  "tmax": "2024-12-31",
  "modifications": [
    { "type": "pumping_synthetic", "usage": "aep", "pattern": "constant", "rate_m3d": 300, "distance_m": 500, "start": "2020-01-01", "end": "2024-12-31", "rfunc": "Exponential", "season_months": [1,2,3,4,5,6,7,8,9,10,11,12], "peak_months": [6,7,8], "peak_factor": 1.25, "pulse_duration_days": 30 }
  ]
}
```

### 6.2 API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/pastas/{run_id}/scenarios` | List saved scenarios for a model |
| `POST` | `/api/v1/pastas/{run_id}/scenarios` | Save a named scenario |
| `DELETE` | `/api/v1/pastas/{run_id}/scenarios/{name}` | Delete a scenario |
| `POST` | `/api/v1/pastas/{run_id}/scenarios/{name}/apply` | Load scenario; body `{ "target_run_id": "..." }` to apply cross-model |

### 6.3 Cross-model reuse

When applying a saved scenario to a different model:
- Modifications are copied as-is
- `tmin`/`tmax` clamped to the target model's observed period if they overflow
- Warnings recalculated for the target aquifer family
- Explicit warning if aquifer family differs between source and target

### 6.4 Frontend UX

- **Save button** next to "Simuler" → dialog with name + optional description
- **"Mes scénarios" section** in sidebar below presets:
  - List of saved scenarios for current model
  - Each entry: name, date, modification count
  - Click to load, trash icon to delete
  - Badge if scenario originated from another model

---

## 7. Schema Changes

### 7.1 Backend Pydantic — modified `PumpingSynthetic`

```python
class PumpingSynthetic(BaseModel):
    type: Literal["pumping_synthetic"]
    usage: Optional[Literal["aep", "irrigation", "industrial"]] = None
    pattern: Literal["constant", "seasonal", "pulse"]
    rate_m3d: float = Field(ge=0)
    start: date
    end: date
    distance_m: float = Field(gt=0)
    screen_depth_m: Optional[float] = None
    rfunc: Literal["Hantush", "Exponential"] = "Exponential"
    period_days: int = 365
    season_months: Optional[list[int]] = None
    peak_months: Optional[list[int]] = None
    peak_factor: Optional[float] = Field(default=None, ge=1.0, le=2.0)
    pulse_duration_days: int = Field(default=30, ge=1)
```

New fields: `usage`, `peak_months`, `peak_factor`. All optional for backward compatibility.

### 7.2 New schemas

- `AquiferFamily` — Literal enum of 5 families
- `PumpingRange`, `PumpingProfile` — referential types
- `ValidateRequest`, `ValidateResponse` — pre-validation
- `ScenarioPresetsResponse` — full referential for frontend cache
- `SaveScenarioRequest` — name + description + modifications
- `SavedScenario` — stored scenario metadata

---

## 8. Files to Create/Modify

### New files
- `dashboard/utils/pastas/scenario_presets.py` — referential, validation, aquifer detection, preset builders
- `frontend/src/hooks/useScenarioPresets.ts` — React Query hook for referential
- `frontend/src/hooks/useSavedScenarios.ts` — CRUD hook for saved scenarios

### Modified files
- `dashboard/utils/pastas/scenario.py` — integrate validation before simulation
- `api/routers/pastas.py` — new endpoints (presets, validate, scenarios CRUD)
- `api/schemas/pastas.py` — new schemas + updated PumpingSynthetic
- `frontend/src/pages/pastas/ScenariosPage.tsx` — new layout with contextual presets + saved scenarios
- `frontend/src/components/pastas/ScenarioComposer.tsx` — usage selector, warnings integration
- `frontend/src/components/pastas/modifications/PumpingSyntheticEditor.tsx` — usage picker, dynamic bounds, inline warnings
- `frontend/src/components/pastas/modifications/ScaleStressEditor.tsx` — hard limits from referential
- `frontend/src/components/pastas/modifications/LinearTrendEditor.tsx` — hard limits from referential
- `frontend/src/components/pastas/ScenarioResultsPanel.tsx` — display enriched warnings
- `frontend/src/lib/api.ts` — new API functions for presets and scenarios

### Test files
- `tests/pastas/test_scenario_presets.py` — referential integrity, validation, detection, preset generation

---

## 9. Backward Compatibility

- `usage` field on `PumpingSynthetic` is `Optional`, default `None` — existing payloads work unchanged
- Existing 4 presets are replaced by 6 contextual presets — no data migration needed
- Existing `/api/v1/pastas/simulate` endpoint unchanged in signature — validation is additive
- Scenarios storage is new (no existing data to migrate)
- `peak_months` and `peak_factor` are optional — existing synthetic pumping patterns unaffected
