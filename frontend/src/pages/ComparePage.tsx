import { useState, useMemo, useEffect } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import Plot from 'react-plotly.js'
import { ArrowLeft, X, Plus, Search, Trash2 } from 'lucide-react'
import { useStationsGeoJSON, usePiezoStationDetail, useHydroStationDetail, usePiezoMonthly, useHydroMonthly, usePiezoSPLI, useHydroSSFI } from '@/hooks/useObservatory'
import { formatNumber, formatDate } from '@/lib/observatory-utils'
import { CLASSIFICATION_COLORS } from '@/lib/observatory-constants'
import { useCompareSelection, MAX_STATIONS } from '@/contexts/CompareSelection'

type StationType = 'piezo' | 'hydro'

// Colorblind-friendly palette (Okabe-Ito) so each station keeps the same color
// across all charts and the comparative table.
const PALETTE = ['#06b6d4', '#f59e0b', '#a78bfa', '#10b981', '#ef4444']

interface Props { stationType: StationType }

// One slot = one set of hooks. We always render exactly MAX_STATIONS slots
// so the hook count stays stable across renders even as the user adds/removes
// stations; inactive slots receive an empty code which short-circuits each
// hook's `enabled` flag.
function StationSlot({ code, type, color, onRemove }: { code: string | null; type: StationType; color: string; onRemove: () => void }) {
  const effective = code ?? ''
  const piezo = usePiezoStationDetail(type === 'piezo' ? effective : '')
  const hydro = useHydroStationDetail(type === 'hydro' ? effective : '')
  const detail: any = type === 'piezo' ? piezo.data : hydro.data
  const name = type === 'piezo' ? (detail?.nom_commune || code) : (detail?.libelle_station || code)
  if (!code) return null
  return (
    <div className="flex items-center gap-2 bg-bg-card border border-white/5 rounded-lg px-3 py-2">
      <span className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: color }} />
      <div className="flex-1 min-w-0">
        <p className="text-xs text-text-primary truncate">{name}</p>
        <p className="text-[10px] text-text-secondary font-mono truncate">{code}</p>
      </div>
      <button onClick={onRemove} aria-label="Retirer" className="p-1 hover:bg-bg-hover rounded">
        <X className="w-3.5 h-3.5 text-text-secondary" />
      </button>
    </div>
  )
}

function StationPicker({ stationType, exclude, onPick }: { stationType: StationType; exclude: string[]; onPick: (code: string) => void }) {
  const { data: geo } = useStationsGeoJSON()
  const [query, setQuery] = useState('')
  const candidates = useMemo(() => {
    if (!geo?.features || !query.trim()) return []
    const q = query.toLowerCase().trim()
    return geo.features
      .filter((f: any) => f.properties.type === stationType)
      .filter((f: any) => !exclude.includes(f.properties.code))
      .filter((f: any) =>
        f.properties.code.toLowerCase().includes(q) ||
        (f.properties.libelle_station || '').toLowerCase().includes(q) ||
        (f.properties.nom_commune || '').toLowerCase().includes(q),
      )
      .slice(0, 8)
  }, [geo, query, stationType, exclude])

  return (
    <div className="space-y-2" data-tour="compare-picker">
      <div className="relative">
        <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-text-secondary" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={`Rechercher une station ${stationType === 'piezo' ? 'piézométrique' : 'hydrométrique'}…`}
          className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg pl-9 pr-3 py-2 text-sm"
        />
      </div>
      {candidates.length > 0 && (
        <ul className="bg-bg-card border border-white/5 rounded-lg divide-y divide-white/5 max-h-48 overflow-y-auto">
          {candidates.map((f: any) => (
            <li key={f.properties.code}>
              <button
                onClick={() => { onPick(f.properties.code); setQuery('') }}
                className="w-full flex items-center gap-2 px-3 py-2 hover:bg-bg-hover text-left transition-colors"
              >
                <Plus className="w-3.5 h-3.5 text-accent-cyan shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-text-primary truncate">{f.properties.libelle_station || f.properties.nom_commune || f.properties.code}</p>
                  <p className="text-[10px] text-text-secondary font-mono truncate">{f.properties.code} · {f.properties.code_departement}</p>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/**
 * Hook-stable wrapper: always calls the same 5 hook tuples. Inactive slots
 * pass an empty code which keeps each query disabled, so this never breaks
 * the rules of hooks no matter how many stations the user picks.
 */
function useCompareSeries(codes: string[], type: StationType) {
  const slots = useMemo(() => {
    const padded = [...codes]
    while (padded.length < MAX_STATIONS) padded.push('')
    return padded.slice(0, MAX_STATIONS)
  }, [codes])

  const piezo0 = usePiezoStationDetail(type === 'piezo' ? slots[0] : '')
  const piezo1 = usePiezoStationDetail(type === 'piezo' ? slots[1] : '')
  const piezo2 = usePiezoStationDetail(type === 'piezo' ? slots[2] : '')
  const piezo3 = usePiezoStationDetail(type === 'piezo' ? slots[3] : '')
  const piezo4 = usePiezoStationDetail(type === 'piezo' ? slots[4] : '')
  const hydro0 = useHydroStationDetail(type === 'hydro' ? slots[0] : '')
  const hydro1 = useHydroStationDetail(type === 'hydro' ? slots[1] : '')
  const hydro2 = useHydroStationDetail(type === 'hydro' ? slots[2] : '')
  const hydro3 = useHydroStationDetail(type === 'hydro' ? slots[3] : '')
  const hydro4 = useHydroStationDetail(type === 'hydro' ? slots[4] : '')
  const details = type === 'piezo' ? [piezo0, piezo1, piezo2, piezo3, piezo4] : [hydro0, hydro1, hydro2, hydro3, hydro4]

  const pm0 = usePiezoMonthly(type === 'piezo' ? slots[0] : '')
  const pm1 = usePiezoMonthly(type === 'piezo' ? slots[1] : '')
  const pm2 = usePiezoMonthly(type === 'piezo' ? slots[2] : '')
  const pm3 = usePiezoMonthly(type === 'piezo' ? slots[3] : '')
  const pm4 = usePiezoMonthly(type === 'piezo' ? slots[4] : '')
  const hm0 = useHydroMonthly(type === 'hydro' ? slots[0] : '')
  const hm1 = useHydroMonthly(type === 'hydro' ? slots[1] : '')
  const hm2 = useHydroMonthly(type === 'hydro' ? slots[2] : '')
  const hm3 = useHydroMonthly(type === 'hydro' ? slots[3] : '')
  const hm4 = useHydroMonthly(type === 'hydro' ? slots[4] : '')
  const monthly = type === 'piezo' ? [pm0, pm1, pm2, pm3, pm4] : [hm0, hm1, hm2, hm3, hm4]

  const ps0 = usePiezoSPLI(type === 'piezo' ? slots[0] : '')
  const ps1 = usePiezoSPLI(type === 'piezo' ? slots[1] : '')
  const ps2 = usePiezoSPLI(type === 'piezo' ? slots[2] : '')
  const ps3 = usePiezoSPLI(type === 'piezo' ? slots[3] : '')
  const ps4 = usePiezoSPLI(type === 'piezo' ? slots[4] : '')
  const hs0 = useHydroSSFI(type === 'hydro' ? slots[0] : '')
  const hs1 = useHydroSSFI(type === 'hydro' ? slots[1] : '')
  const hs2 = useHydroSSFI(type === 'hydro' ? slots[2] : '')
  const hs3 = useHydroSSFI(type === 'hydro' ? slots[3] : '')
  const hs4 = useHydroSSFI(type === 'hydro' ? slots[4] : '')
  const drought = type === 'piezo' ? [ps0, ps1, ps2, ps3, ps4] : [hs0, hs1, hs2, hs3, hs4]

  return slots.map((code, i) => ({
    code,
    detail: details[i].data as any,
    monthly: (monthly[i].data as any[]) ?? [],
    drought: (drought[i].data as any[]) ?? [],
  })).filter((s) => s.code !== '')
}

export default function ComparePage({ stationType }: Props) {
  const [searchParams, setSearchParams] = useSearchParams()
  const selection = useCompareSelection()

  // Codes for this page are the subset of the selection that matches this
  // type. The URL `?codes=…` query lets users share a comparison via link —
  // on mount we hydrate the context with any URL codes that aren't already
  // there.
  const codes = useMemo(
    () => selection.items.filter((i) => i.type === stationType).map((i) => i.code),
    [selection.items, stationType],
  )

  // One-shot URL → context hydration (only when context is empty for this type).
  useEffect(() => {
    const urlCodes = (searchParams.get('codes') ?? '').split(',').filter(Boolean)
    if (urlCodes.length === 0) return
    if (codes.length > 0) return
    // If the context holds the other type, ignore the URL; the user explicitly
    // navigated to a fresh type.
    if (selection.type && selection.type !== stationType) return
    for (const c of urlCodes.slice(0, MAX_STATIONS)) {
      selection.add({ code: c, type: stationType })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Keep URL in sync as the user edits the selection.
  useEffect(() => {
    const desired = codes.join(',')
    const current = searchParams.get('codes') ?? ''
    if (desired === current) return
    const next = new URLSearchParams(searchParams)
    if (desired) next.set('codes', desired)
    else next.delete('codes')
    setSearchParams(next, { replace: true })
  }, [codes, searchParams, setSearchParams])

  const add = (c: string) => selection.add({ code: c, type: stationType })
  const remove = (c: string) => selection.remove(c)

  const series = useCompareSeries(codes, stationType)

  const valueKey = stationType === 'piezo' ? 'niveau_moyen' : 'resultat_moyen'
  const valueLabel = stationType === 'piezo' ? 'Niveau piézométrique (m NGF)' : 'Débit moyen (m³/s)'
  const droughtKey = stationType === 'piezo' ? 'spli' : 'ssfi'
  const droughtLabel = stationType === 'piezo' ? 'Indice piézométrique standardisé (SPLI/IPS)' : 'Indice standardisé de débit (SSFI)'

  const traces = series.map((s, i) => ({
    x: s.monthly.map((m: any) => m.mois),
    y: s.monthly.map((m: any) => m[valueKey]),
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: (stationType === 'piezo' ? s.detail?.nom_commune : s.detail?.libelle_station) || s.code,
    line: { color: PALETTE[i % PALETTE.length], width: 1.5 },
  }))

  const droughtTraces = series.map((s, i) => ({
    x: s.drought.map((d: any) => d.mois),
    y: s.drought.map((d: any) => d[droughtKey]),
    type: 'scatter' as const,
    mode: 'lines' as const,
    name: (stationType === 'piezo' ? s.detail?.nom_commune : s.detail?.libelle_station) || s.code,
    line: { color: PALETTE[i % PALETTE.length], width: 1.5 },
  }))

  const baseLayout = {
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#9ca3af', size: 11 },
    margin: { t: 10, r: 20, b: 40, l: 55 },
    height: 320,
    xaxis: { type: 'date' as const, gridcolor: 'rgba(255,255,255,0.04)' },
    yaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
    legend: { orientation: 'h' as const, y: -0.22, font: { size: 10, color: '#9ca3af' } },
    hovermode: 'x unified' as const,
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-7xl mx-auto px-6 py-6 space-y-5">
        <div className="flex items-center gap-4">
          <Link to="/" className="p-2 hover:bg-bg-hover rounded-lg" aria-label="Retour à l'observatoire">
            <ArrowLeft className="w-5 h-5 text-text-secondary" />
          </Link>
          <div>
            <p className="text-xs text-accent-cyan font-medium uppercase tracking-wide">Comparaison</p>
            <h1 className="text-xl font-bold text-text-primary">
              {stationType === 'piezo' ? 'Stations piézométriques' : 'Stations hydrométriques'}
            </h1>
          </div>
        </div>

        <section className="bg-gray-900/50 rounded-xl border border-white/5 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-300">Stations sélectionnées ({codes.length}/{MAX_STATIONS})</h2>
            <div className="flex items-center gap-3">
              {codes.length > 0 && (
                <button
                  onClick={() => selection.clear()}
                  className="flex items-center gap-1 text-[11px] text-text-secondary hover:text-red-400 transition-colors"
                  title="Vider la sélection"
                >
                  <Trash2 className="w-3 h-3" />
                  Vider
                </button>
              )}
              <Link
                to={stationType === 'piezo' ? '/compare/hydro' : '/compare/piezo'}
                className="text-xs text-accent-cyan hover:underline"
              >
                Basculer vers {stationType === 'piezo' ? 'hydro' : 'piézo'}
              </Link>
            </div>
          </div>
          {codes.length === 0 ? (
            <p className="text-xs text-text-secondary italic">Aucune station sélectionnée.</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {codes.map((c, i) => (
                <StationSlot
                  key={c}
                  code={c}
                  type={stationType}
                  color={PALETTE[i % PALETTE.length]}
                  onRemove={() => remove(c)}
                />
              ))}
            </div>
          )}
          {codes.length < MAX_STATIONS && (
            <StationPicker stationType={stationType} exclude={codes} onPick={add} />
          )}
        </section>

        {codes.length < 2 ? (
          <div className="bg-bg-card border border-white/5 rounded-xl p-8 text-center text-text-secondary text-sm">
            Ajoutez au moins 2 stations pour démarrer la comparaison.
          </div>
        ) : (
          <>
            <section className="bg-bg-card border border-white/5 rounded-xl p-5" data-tour="compare-chart">
              <h2 className="text-sm font-semibold text-text-primary mb-3">{valueLabel}</h2>
              <Plot
                data={traces}
                layout={{ ...baseLayout, yaxis: { ...baseLayout.yaxis, title: { text: valueLabel } } }}
                config={{ displayModeBar: false }}
                useResizeHandler
                className="w-full"
              />
            </section>

            <section className="bg-bg-card border border-white/5 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-text-primary mb-3">{droughtLabel}</h2>
              <Plot
                data={droughtTraces}
                layout={{ ...baseLayout, yaxis: { ...baseLayout.yaxis, title: { text: droughtLabel }, zeroline: true, zerolinecolor: 'rgba(255,255,255,0.15)' } }}
                config={{ displayModeBar: false }}
                useResizeHandler
                className="w-full"
              />
            </section>

            <section className="bg-bg-card border border-white/5 rounded-xl p-5">
              <h2 className="text-sm font-semibold text-text-primary mb-3">Statistiques comparatives</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/5">
                      <th className="px-3 py-2 text-left text-text-secondary font-medium">Station</th>
                      <th className="px-3 py-2 text-right text-text-secondary font-medium">Moyenne</th>
                      <th className="px-3 py-2 text-right text-text-secondary font-medium">Min</th>
                      <th className="px-3 py-2 text-right text-text-secondary font-medium">Max</th>
                      <th className="px-3 py-2 text-left text-text-secondary font-medium">Dernière mesure</th>
                      <th className="px-3 py-2 text-left text-text-secondary font-medium">Classification</th>
                    </tr>
                  </thead>
                  <tbody>
                    {series.map((s, i) => {
                      const d = s.detail
                      if (!d) {
                        return (
                          <tr key={s.code} className="border-b border-white/5">
                            <td colSpan={6} className="px-3 py-2 text-text-secondary text-xs italic">Chargement de {s.code}…</td>
                          </tr>
                        )
                      }
                      const mean = stationType === 'piezo' ? d.niveau_moyen_global : d.resultat_moyen_global
                      const min = stationType === 'piezo' ? d.niveau_min_absolu : d.resultat_min_global
                      const max = stationType === 'piezo' ? d.niveau_max_absolu : d.resultat_max_global
                      const last = d.derniere_mesure
                      const cls = stationType === 'piezo' ? d.classification_derniere_annee : d.classification_resultat_dern_annee
                      const clsColor = cls ? CLASSIFICATION_COLORS[cls] ?? '#6b7280' : null
                      const name = (stationType === 'piezo' ? d.nom_commune : d.libelle_station) || s.code
                      return (
                        <tr key={s.code} className="border-b border-white/5 hover:bg-bg-hover">
                          <td className="px-3 py-2">
                            <div className="flex items-center gap-2">
                              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: PALETTE[i % PALETTE.length] }} />
                              <Link to={`/station/${stationType}/${s.code}`} className="text-accent-cyan hover:underline">
                                {name}
                              </Link>
                            </div>
                          </td>
                          <td className="px-3 py-2 text-right font-mono text-text-primary">{formatNumber(mean, 2)}</td>
                          <td className="px-3 py-2 text-right font-mono text-text-secondary">{formatNumber(min, 2)}</td>
                          <td className="px-3 py-2 text-right font-mono text-text-secondary">{formatNumber(max, 2)}</td>
                          <td className="px-3 py-2 text-text-secondary">{formatDate(last)}</td>
                          <td className="px-3 py-2">
                            {clsColor && (
                              <span className="inline-flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: clsColor }} />
                                <span className="text-text-secondary">{(cls ?? '').replace(/_/g, ' ').toLowerCase()}</span>
                              </span>
                            )}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  )
}
