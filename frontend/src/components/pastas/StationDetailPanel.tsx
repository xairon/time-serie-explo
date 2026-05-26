import { Loader2, TrendingDown, TrendingUp, Minus, MapPin, Droplets, Thermometer, Mountain } from 'lucide-react'
import Plot from 'react-plotly.js'
import type { PastasStationPreview } from '@/lib/types'
import { InfoTip } from './InfoTip'

const BDLISA_LABELS: Record<string, string> = {
  '0': 'Socle', '3': 'Alluvial', '4': 'Karst', '5': 'Sédimentaire',
  '6': 'Volcanique', '7': 'Montagne',
}
const MILIEU_LABELS: Record<string, string> = {
  '1': 'poreux', '2': 'fissuré', '3': 'karstique', '4': 'double porosité',
  '5': 'alluvial', '8': 'composite',
}
const BDLISA_COLORS: Record<string, string> = {
  '0': '#78716c', '3': '#22d3ee', '4': '#f97316', '5': '#60a5fa',
  '6': '#ef4444', '7': '#a78bfa',
}

const TREND_CONFIG: Record<string, { label: string; color: string; Icon: React.ElementType }> = {
  hausse: { label: 'En hausse', color: 'text-green-400', Icon: TrendingUp },
  baisse: { label: 'En baisse', color: 'text-red-400', Icon: TrendingDown },
  stable: { label: 'Stable', color: 'text-text-secondary', Icon: Minus },
}

const ALERT_COLORS: Record<string, string> = {
  EXTREMEMENT_BAS: 'bg-red-500', TRES_BAS: 'bg-orange-500', BAS: 'bg-yellow-500',
  HAUT: 'bg-blue-400', TRES_HAUT: 'bg-blue-600', EXTREMEMENT_HAUT: 'bg-purple-500',
}

const CLASS_COLORS: Record<string, string> = {
  tres_bas: 'text-red-400 border-red-500/30 bg-red-500/10',
  bas: 'text-orange-400 border-orange-500/30 bg-orange-500/10',
  modere_bas: 'text-yellow-400 border-yellow-500/30 bg-yellow-500/10',
  autour_moyenne: 'text-text-secondary border-white/10 bg-white/5',
  modere_haut: 'text-blue-300 border-blue-500/30 bg-blue-500/10',
  haut: 'text-blue-400 border-blue-500/30 bg-blue-500/10',
  tres_haut: 'text-purple-400 border-purple-500/30 bg-purple-500/10',
}

function KpiCard({ label, value, unit, icon: Icon, sub, tip }: {
  label: string; value: string | null; unit?: string; icon?: React.ElementType; sub?: string; tip?: string
}) {
  return (
    <div className="bg-bg-primary rounded-lg border border-white/5 p-3">
      <div className="flex items-center gap-1.5 mb-1">
        {Icon && <Icon className="w-3.5 h-3.5 text-text-muted" />}
        <span className="text-[10px] text-text-muted uppercase tracking-wide">{label}</span>
        {tip && <InfoTip text={tip} iconSize={10} />}
      </div>
      <div className="text-lg font-semibold text-text-primary">
        {value ?? '—'}
        {unit && value && <span className="text-xs text-text-muted ml-1">{unit}</span>}
      </div>
      {sub && <div className="text-[10px] text-text-muted mt-0.5">{sub}</div>}
    </div>
  )
}

interface Props {
  stationInfo: Record<string, unknown> | undefined
  stationInfoLoading: boolean
  preview: PastasStationPreview | undefined
  previewLoading: boolean
  onRangeChange?: (tmin: string, tmax: string) => void
}

export function StationDetailPanel({ stationInfo, stationInfoLoading, preview, previewLoading, onRangeChange }: Props) {
  if (stationInfoLoading) {
    return (
      <div className="flex items-center justify-center h-32 text-text-muted text-sm gap-2">
        <Loader2 className="w-4 h-4 animate-spin" /> Chargement des informations de la station…
      </div>
    )
  }

  if (!stationInfo) return null

  const s = stationInfo
  const trendKey = (s.tendance_classification as string)?.toLowerCase() ?? ''
  const trendCfg = TREND_CONFIG[trendKey]
  const alertLevel = s.niveau_alerte as string | null
  const classif = s.classification_derniere_annee as string | null
  const natureEh = s.nature_eh as string | undefined
  const milieuEh = s.milieu_eh as string | undefined
  const bdlisaLabel = natureEh ? (BDLISA_LABELS[natureEh] ?? `Type ${natureEh}`) : null
  const milieuLabel = milieuEh ? MILIEU_LABELS[milieuEh] : null
  const bdlisaColor = natureEh ? (BDLISA_COLORS[natureEh] ?? '#6b7280') : '#6b7280'
  const slope = s.slope_niveau as number | null
  const pctile = s.percentile_derniere_annee as number | null

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-base font-mono font-semibold text-accent-cyan">{s.code_bss as string}</span>
            {bdlisaLabel && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border"
                style={{ borderColor: `${bdlisaColor}40`, backgroundColor: `${bdlisaColor}15`, color: bdlisaColor }}>
                {bdlisaLabel}{milieuLabel ? ` (${milieuLabel})` : ''}
              </span>
            )}
            {alertLevel && (
              <span className={`w-2.5 h-2.5 rounded-full ${ALERT_COLORS[alertLevel] ?? 'bg-gray-400'}`}
                title={`Alerte : ${alertLevel}`} />
            )}
          </div>
          <div className="flex items-center gap-2 mt-1 text-xs text-text-muted">
            <MapPin className="w-3 h-3" />
            {s.nom_commune as string} ({s.code_departement as string} — {s.nom_departement as string})
            {s.altitude != null && <span>· {(s.altitude as number).toFixed(0)} m NGF</span>}
          </div>
        </div>
        {trendCfg && (
          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-white/10 ${trendCfg.color}`}>
            <trendCfg.Icon className="w-4 h-4" />
            <span className="text-xs font-medium">{trendCfg.label}</span>
            {slope != null && <span className="text-[10px] text-text-muted">({slope > 0 ? '+' : ''}{(slope * 100).toFixed(1)} cm/an)</span>}
            <InfoTip text="Tendance long terme issue d'une régression linéaire sur l'ensemble de la chronique. Positive = nappe en hausse, négative = en baisse. La pente (cm/an) quantifie le rythme. Une baisse peut indiquer une augmentation des prélèvements ou une diminution de la recharge." iconSize={10} />
          </div>
        )}
      </div>

      {/* Classification + percentile bar */}
      {(classif || pctile != null) && (
        <div className="flex items-center gap-3">
          {classif && (
            <span className={`px-2 py-0.5 rounded text-[10px] font-medium border ${CLASS_COLORS[classif] ?? 'text-text-muted border-white/10'}`}>
              {classif.replace(/_/g, ' ')}
            </span>
          )}
          {pctile != null && (
            <div className="flex-1 flex items-center gap-2">
              <span className="text-[10px] text-text-muted shrink-0 flex items-center gap-1">
                Percentile dernière année
                <InfoTip text="Positionnement du niveau moyen de la dernière année parmi toutes les années historiques. 0% = l'année la plus basse jamais enregistrée, 100% = la plus haute. Détermine la classification (très bas < 10%, bas 10-25%, modérément bas 25-40%, autour de la moyenne 40-60%, modérément haut 60-75%, haut 75-90%, très haut > 90%)." iconSize={10} />
              </span>
              <div className="flex-1 h-2 bg-bg-primary rounded-full overflow-hidden border border-white/5">
                <div className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.min(100, Math.max(0, pctile * 100))}%`,
                    backgroundColor: pctile < 0.1 ? '#ef4444' : pctile < 0.25 ? '#f97316' : pctile > 0.75 ? '#3b82f6' : '#22d3ee',
                  }} />
              </div>
              <span className="text-[10px] font-mono text-text-muted">{(pctile * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
      )}

      {/* KPI cards */}
      <div className="grid grid-cols-4 gap-2">
        <KpiCard label="Niveau moyen" value={s.niveau_moyen_global != null ? (s.niveau_moyen_global as number).toFixed(2) : null} unit="m"
          icon={Droplets}
          tip="Niveau d'eau moyen historique (m NGF) sur l'ensemble des mesures. Min/Max indiquent les extrêmes absolus jamais enregistrés."
          sub={s.niveau_min_absolu != null ? `Min ${(s.niveau_min_absolu as number).toFixed(1)} — Max ${(s.niveau_max_absolu as number).toFixed(1)} m` : undefined} />
        <KpiCard label="Amplitude" value={s.amplitude_totale != null ? (s.amplitude_totale as number).toFixed(2) : null} unit="m"
          icon={Mountain}
          tip="Plage totale (max − min) du niveau d'eau. σ est l'écart-type — une mesure de la variation typique. Une grande amplitude avec un σ faible suggère des événements extrêmes rares."
          sub={s.niveau_stddev_global != null ? `σ = ${(s.niveau_stddev_global as number).toFixed(2)} m` : undefined} />
        <KpiCard label="Chronique" value={s.nb_mesures_total != null ? String(s.nb_mesures_total) : null} unit="obs"
          tip="Nombre total de mesures et étendue temporelle. Des chroniques longues (15+ ans) fournissent une calibration plus fiable en capturant les cycles secs et humides."
          sub={s.premiere_mesure && s.derniere_mesure ? `${(s.premiere_mesure as string).slice(0, 4)}–${(s.derniere_mesure as string).slice(0, 4)}` : undefined} />
        <KpiCard label="Climat" value={s.temperature_moyenne_globale != null ? (s.temperature_moyenne_globale as number).toFixed(1) : null} unit="°C"
          icon={Thermometer}
          tip="Température moyenne et précipitations mensuelles issues de la réanalyse ERA5 au point de grille le plus proche. Elles alimentent le stress de recharge dans le modèle Pastas."
          sub={s.precipitation_moyenne_mensuelle != null ? `${(s.precipitation_moyenne_mensuelle as number).toFixed(0)} mm/mois moy.` : undefined} />
      </div>

      {/* Time series chart (loaded async) */}
      {previewLoading && (
        <div className="flex items-center justify-center h-24 text-text-muted text-sm gap-2 bg-bg-card rounded-lg border border-white/5">
          <Loader2 className="w-4 h-4 animate-spin" /> Chargement de la série temporelle…
        </div>
      )}

      {preview && !previewLoading && (
        <div className="bg-bg-card rounded-lg border border-white/5 p-2">
          <Plot
            data={[
              {
                x: preview.piezo.index, y: preview.piezo.values,
                name: 'Piézo (m)', type: 'scatter', mode: 'lines',
                line: { color: '#60a5fa', width: 1 }, xaxis: 'x', yaxis: 'y',
              },
              {
                x: preview.precip.index, y: preview.precip.values,
                name: 'Précip. (mm/j)', type: 'bar',
                marker: { color: 'rgba(59,130,246,0.3)' }, xaxis: 'x', yaxis: 'y2',
              },
              {
                x: preview.evap.index, y: preview.evap.values,
                name: 'ETP (mm/j)', type: 'scatter', mode: 'lines',
                line: { color: '#f97316', width: 1 }, xaxis: 'x', yaxis: 'y3',
              },
            ]}
            layout={{
              paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
              font: { color: '#9ca3af', size: 10 },
              margin: { t: 10, r: 20, b: 40, l: 50 },
              height: 320, showlegend: false,
              grid: { rows: 3, columns: 1, subplots: ['xy', 'xy2', 'xy3'], roworder: 'top to bottom' as const },
              xaxis: { gridcolor: 'rgba(255,255,255,0.03)', rangeslider: { visible: true, thickness: 0.06 } },
              yaxis: { title: { text: 'Piézo (m)' }, gridcolor: 'rgba(255,255,255,0.05)', domain: [0.7, 1] },
              yaxis2: { title: { text: 'P (mm/j)' }, gridcolor: 'rgba(255,255,255,0.05)', domain: [0.38, 0.65] },
              yaxis3: { title: { text: 'ETP (mm/j)' }, gridcolor: 'rgba(255,255,255,0.05)', domain: [0.0, 0.3] },
            }}
            useResizeHandler className="w-full"
            onRelayout={(e: Record<string, unknown>) => {
              if (onRangeChange && e['xaxis.range[0]'] && e['xaxis.range[1]']) {
                onRangeChange(String(e['xaxis.range[0]']).slice(0, 10), String(e['xaxis.range[1]']).slice(0, 10))
              }
            }}
          />
        </div>
      )}
    </div>
  )
}
