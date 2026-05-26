import { useState } from 'react'
import { Layers, X, RotateCcw, ChevronDown, ChevronRight } from 'lucide-react'
import { CLASSIFICATION_ORDER, CLASSIFICATION_LABELS, CLASSIFICATION_COLORS } from '@/lib/observatory-constants'
import type { ObsFilters } from '@/hooks/useObservatory'

const ZONE_LAYERS: { id: string; label: string; group: string; color?: string }[] = [
  { id: 'regions', label: 'Régions', group: 'Administratif' },
  { id: 'depts', label: 'Départements', group: 'Administratif' },
  { id: 'bassins', label: 'Bassins (SANDRE)', group: 'Administratif' },
  { id: 'her', label: 'Hydroécorégions (HER-2)', group: 'Administratif', color: '#34d399' },
  { id: 'region-hydro', label: 'Régions hydrographiques', group: 'Zonage SANDRE', color: '#3b82f6' },
  { id: 'secteur-hydro', label: 'Secteurs hydrographiques', group: 'Zonage SANDRE', color: '#06b6d4' },
  { id: 'sous-secteur-hydro', label: 'Sous-secteurs hydro', group: 'Zonage SANDRE', color: '#14b8a6' },
  { id: 'zone-hydro', label: 'Zones hydrographiques', group: 'Zonage SANDRE', color: '#10b981' },
]

const OVERLAY_LAYERS: { id: string; label: string; group: string; color: string }[] = [
  { id: 'cours-eau-1', label: 'Cours d\'eau majeurs (>100km)', group: 'Réseau hydrographique', color: '#60a5fa' },
  { id: 'cours-eau-2', label: 'Cours d\'eau secondaires', group: 'Réseau hydrographique', color: '#93c5fd' },
  { id: 'plan-eau', label: 'Plans d\'eau', group: 'Réseau hydrographique', color: '#38bdf8' },
  { id: 'masse-eau-riv', label: 'Masses d\'eau (DCE)', group: 'Hydro-écologie', color: '#c084fc' },
]

interface Props {
  showPiezo: boolean; setShowPiezo: (v: boolean) => void
  showHydro: boolean; setShowHydro: (v: boolean) => void
  showExcluded: boolean; setShowExcluded: (v: boolean) => void
  showTerrain: boolean; setShowTerrain: (v: boolean) => void
  filters: ObsFilters; setFilter: (key: string, value: string | string[] | undefined) => void
  filteredPiezo?: number; totalPiezo?: number; filteredHydro?: number; totalHydro?: number
  activeZoneLayer: string | null; onZoneLayerChange: (id: string | null) => void
  overlayLayers: Set<string>; onOverlayToggle: (id: string) => void
  onResetSpatial?: () => void; hasSpatialFilter?: boolean
}

function AccordionSection({ id, title, badge, defaultOpen, children }: { id: string; title: string; badge?: string; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen ?? false)
  return (
    <div className="border-b border-white/5 last:border-0">
      <button onClick={() => setOpen(v => !v)} className="flex items-center justify-between w-full px-4 py-3 hover:bg-bg-hover transition-colors" aria-expanded={open} aria-controls={`section-${id}`}>
        <span className="text-sm font-medium text-text-primary">{title}</span>
        <div className="flex items-center gap-2">
          {badge && <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-accent-cyan/20 text-accent-cyan font-mono">{badge}</span>}
          {open ? <ChevronDown className="w-4 h-4 text-text-secondary" /> : <ChevronRight className="w-4 h-4 text-text-secondary" />}
        </div>
      </button>
      <div id={`section-${id}`} className="grid transition-all duration-200" style={{ gridTemplateRows: open ? '1fr' : '0fr' }}>
        <div className="overflow-hidden"><div className="px-4 pb-3">{children}</div></div>
      </div>
    </div>
  )
}

export function RightDrawer(props: Props) {
  const [drawerOpen, setDrawerOpen] = useState(false)

  const hasActiveFilter = props.filters.activeOnly === false || props.filters.minObservations != null || props.filters.lastMeasurementAfter != null || (props.filters.classification != null && props.filters.classification.length > 0) || props.filters.codeDepartement != null || props.filters.showFiable === false || props.filters.showIndicatif === true || props.filters.showInsuffisant === true || props.hasSpatialFilter

  const resetFilters = () => {
    ;['active_only', 'min_obs', 'last_after', 'classif', 'dept', 'bdlisa', 'bassin', 'fiable', 'indicatif', 'insuffisant'].forEach(k => props.setFilter(k, undefined))
    props.onResetSpatial?.()
  }

  const zoneGroups = ZONE_LAYERS.reduce<{ label: string; layers: typeof ZONE_LAYERS[number][] }[]>((acc, l) => { let g = acc.find(x => x.label === l.group); if (!g) { g = { label: l.group, layers: [] }; acc.push(g) }; g.layers.push(l); return acc }, [])
  const overlayGroups = OVERLAY_LAYERS.reduce<{ label: string; layers: typeof OVERLAY_LAYERS[number][] }[]>((acc, l) => { let g = acc.find(x => x.label === l.group); if (!g) { g = { label: l.group, layers: [] }; acc.push(g) }; g.layers.push(l); return acc }, [])

  return (
    <>
      <button onClick={() => setDrawerOpen(v => !v)} aria-label={drawerOpen ? 'Fermer le panneau' : 'Ouvrir le panneau'} className={`absolute top-4 right-4 z-20 p-2.5 rounded-lg border transition-colors ${drawerOpen ? 'bg-accent-cyan/20 border-accent-cyan/30 text-accent-cyan' : 'bg-bg-card/90 backdrop-blur-md border-white/10 text-text-secondary hover:text-text-primary'}`}>
        <Layers className="w-5 h-5" />
      </button>
      {drawerOpen && <div className="md:hidden fixed inset-0 bg-black/50 backdrop-blur-sm z-30" onClick={() => setDrawerOpen(false)} />}
      <div className={`absolute top-0 right-0 h-full z-30 w-full sm:w-80 bg-bg-card border-l border-white/10 shadow-2xl transition-transform duration-200 ease-out overflow-y-auto ${drawerOpen ? 'translate-x-0' : 'translate-x-full pointer-events-none'}`}>
        <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
          <h2 className="text-sm font-semibold text-text-primary">Panneau de contrôle</h2>
          <button onClick={() => setDrawerOpen(false)} className="p-1 hover:bg-bg-hover rounded" aria-label="Fermer"><X className="w-4 h-4 text-text-secondary" /></button>
        </div>

        <AccordionSection id="data" title="Données" defaultOpen>
          <div className="flex gap-2">
            <button onClick={() => props.setShowPiezo(!props.showPiezo)} aria-pressed={props.showPiezo} className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-colors border ${props.showPiezo ? 'bg-accent-cyan/20 text-accent-cyan border-accent-cyan/30' : 'bg-bg-primary text-text-secondary border-white/10'}`}>Piézométrie</button>
            <button onClick={() => props.setShowHydro(!props.showHydro)} aria-pressed={props.showHydro} className={`flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-colors border ${props.showHydro ? 'bg-accent-indigo/20 text-accent-indigo border-accent-indigo/30' : 'bg-bg-primary text-text-secondary border-white/10'}`}>Hydrométrie</button>
            <div className="flex flex-col gap-1 mt-2">
              <label className="flex items-center gap-2 cursor-pointer group"><input type="checkbox" checked={props.showExcluded} onChange={() => props.setShowExcluded(!props.showExcluded)} className="w-3.5 h-3.5 accent-gray-500 rounded" /><span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">Stations filtrées <span className="text-text-secondary/60">(gris)</span></span></label>
              <label className="flex items-center gap-2 cursor-pointer group"><input type="checkbox" checked={props.showTerrain} onChange={() => props.setShowTerrain(!props.showTerrain)} className="w-3.5 h-3.5 accent-amber-600 rounded" /><span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">Relief <span className="text-text-secondary/60">(ombrage)</span></span></label>
            </div>
          </div>
        </AccordionSection>

        <AccordionSection id="filters" title="Filtres" badge={(() => {
          const parts: string[] = []
          if (props.showPiezo && props.filteredPiezo != null && props.totalPiezo != null) parts.push(props.filteredPiezo !== props.totalPiezo ? `${props.filteredPiezo.toLocaleString()}/${props.totalPiezo.toLocaleString()} piézo` : `${props.totalPiezo.toLocaleString()} piézo`)
          if (props.showHydro && props.filteredHydro != null && props.totalHydro != null) parts.push(props.filteredHydro !== props.totalHydro ? `${props.filteredHydro.toLocaleString()}/${props.totalHydro.toLocaleString()} hydro` : `${props.totalHydro.toLocaleString()} hydro`)
          return parts.length > 0 ? parts.join(' - ') : undefined
        })()}>
          <div className="space-y-3">
            <div><label className="text-xs text-text-secondary block mb-2">Qualité des données</label>
              <div className="space-y-1">
                <label className="flex items-center gap-2 cursor-pointer group"><input type="checkbox" checked={props.filters.showFiable ?? true} onChange={(e) => props.setFilter('fiable', e.target.checked ? undefined : 'false')} className="w-3.5 h-3.5 accent-accent-cyan rounded" /><span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">Fiable <span className="text-text-secondary/60">(10+ ans)</span></span></label>
                <label className="flex items-center gap-2 cursor-pointer group"><input type="checkbox" checked={props.filters.showIndicatif ?? false} onChange={(e) => props.setFilter('indicatif', e.target.checked ? 'true' : undefined)} className="w-3.5 h-3.5 accent-yellow-500 rounded" /><span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">Indicative <span className="text-text-secondary/60">(5-9 ans)</span></span></label>
                <label className="flex items-center gap-2 cursor-pointer group"><input type="checkbox" checked={props.filters.showInsuffisant ?? false} onChange={(e) => props.setFilter('insuffisant', e.target.checked ? 'true' : undefined)} className="w-3.5 h-3.5 accent-red-500 rounded" /><span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">Insuffisante <span className="text-text-secondary/60">(&lt; 5 ans)</span></span></label>
              </div>
            </div>
            <label className="flex items-center gap-2 cursor-pointer group"><input type="checkbox" checked={props.filters.activeOnly ?? true} onChange={(e) => props.setFilter('active_only', e.target.checked ? undefined : 'false')} className="w-3.5 h-3.5 accent-accent-cyan rounded" /><span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">Données de l'année en cours uniquement</span></label>
            <div><label className="text-xs text-text-secondary block mb-1">Département</label><input type="text" value={props.filters.codeDepartement ?? ''} onChange={(e) => props.setFilter('dept', e.target.value || undefined)} placeholder="ex. 75" className="w-full px-2.5 py-1.5 bg-bg-primary border border-white/10 rounded text-sm text-text-primary focus:outline-none focus:border-accent-cyan/50" /></div>
            <div><label className="text-xs text-text-secondary block mb-2">Classification</label>
              <div className="flex flex-wrap gap-1.5">
                {CLASSIFICATION_ORDER.map((cls) => {
                  const active = props.filters.classification?.includes(cls) ?? false; const color = CLASSIFICATION_COLORS[cls]
                  return (<button key={cls} onClick={() => { const current = props.filters.classification ?? []; const next = active ? current.filter(c => c !== cls) : [...current, cls]; props.setFilter('classif', next.length > 0 ? next : undefined) }} aria-pressed={active} className="px-2 py-1 rounded text-xs font-medium transition-colors border" style={{ backgroundColor: active ? `${color}30` : 'transparent', borderColor: active ? color : 'rgba(255,255,255,0.1)', color: active ? color : '#9ca3af' }}>{CLASSIFICATION_LABELS[cls]}</button>)
                })}
              </div>
            </div>
            <div><label className="text-xs text-text-secondary block mb-1">Observations min. (jours)</label><input type="number" value={props.filters.minObservations ?? ''} onChange={(e) => props.setFilter('min_obs', e.target.value || undefined)} placeholder="500" className="w-full px-2.5 py-1.5 bg-bg-primary border border-white/10 rounded text-sm text-text-primary focus:outline-none focus:border-accent-cyan/50" /></div>
            <div><label className="text-xs text-text-secondary block mb-1">Dernière mesure après</label><input type="date" value={props.filters.lastMeasurementAfter ?? ''} onChange={(e) => props.setFilter('last_after', e.target.value || undefined)} className="w-full px-2.5 py-1.5 bg-bg-primary border border-white/10 rounded text-sm text-text-primary focus:outline-none focus:border-accent-cyan/50" /></div>
            {props.hasSpatialFilter && (<div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-accent-cyan/10 border border-accent-cyan/20 text-accent-cyan text-xs"><span>Filtre spatial actif (zone sélectionnée)</span><button onClick={() => props.onResetSpatial?.()} className="ml-auto hover:text-white transition-colors" aria-label="Retirer le filtre spatial"><X className="w-3 h-3" /></button></div>)}
            {hasActiveFilter && (<button onClick={resetFilters} className="flex items-center gap-1.5 w-full justify-center px-3 py-2 rounded-lg text-xs font-medium text-text-secondary hover:text-text-primary bg-white/5 hover:bg-white/10 border border-white/10 transition-colors"><RotateCcw className="w-3 h-3" />Réinitialiser</button>)}
          </div>
        </AccordionSection>

        <AccordionSection id="layers" title="Couches">
          {zoneGroups.map(group => (<div key={group.label} className="mb-3"><span className="text-[10px] font-semibold text-white/50 uppercase tracking-wider block mb-1">{group.label}</span>{group.layers.map(layer => { const active = props.activeZoneLayer === layer.id; return (<label key={layer.id} className="flex items-center gap-2 py-1 cursor-pointer group"><input type="checkbox" checked={active} onChange={() => props.onZoneLayerChange(active ? null : layer.id)} className="w-3.5 h-3.5 accent-accent-cyan rounded" />{layer.color && <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: layer.color }} />}<span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">{layer.label}</span></label>) })}</div>))}
          {overlayGroups.map(group => (<div key={group.label} className="mb-3"><span className="text-[10px] font-semibold text-white/50 uppercase tracking-wider block mb-1">{group.label}</span>{group.layers.map(layer => (<label key={layer.id} className="flex items-center gap-2 py-1 cursor-pointer group"><input type="checkbox" checked={props.overlayLayers.has(layer.id)} onChange={() => props.onOverlayToggle(layer.id)} className="w-3.5 h-3.5 accent-accent-cyan rounded" /><span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: layer.color }} /><span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">{layer.label}</span></label>))}</div>))}
        </AccordionSection>
      </div>
    </>
  )
}
