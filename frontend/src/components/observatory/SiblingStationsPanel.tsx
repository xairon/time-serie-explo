import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Waves, Info } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { usePiezoSiblings, useHydroSiblings } from '@/hooks/useObservatory'
import { CLASSIFICATION_COLORS, CLASSIFICATION_LABELS, CLASSIFICATION_ORDER } from '@/lib/observatory-constants'

type Props = {
  code: string
  type: 'piezo' | 'hydro'
  variant?: 'page' | 'drawer'
}

type Row = { to: string; title: string; subtitle?: string; classification: string | null }

export function SiblingStationsPanel({ code, type, variant = 'page' }: Props) {
  const { t } = useTranslation()
  const isPiezo = type === 'piezo'
  const [piezoLevel, setPiezoLevel] = useState<'nappe' | 'systeme'>('nappe')
  const [hydroLevel, setHydroLevel] = useState<'site' | 'cours_eau'>('site')

  const piezo = usePiezoSiblings(isPiezo ? code : '', piezoLevel)
  const hydro = useHydroSiblings(!isPiezo ? code : '', hydroLevel)

  const levels: { value: string; label: string; hint: string }[] = isPiezo
    ? [
        { value: 'nappe', label: t('observatory.siblings.piezo.nappe'), hint: t('observatory.siblings.piezo.nappeHint') },
        { value: 'systeme', label: t('observatory.siblings.piezo.systeme'), hint: t('observatory.siblings.piezo.systemeHint') },
      ]
    : [
        { value: 'site', label: t('observatory.siblings.hydro.site'), hint: t('observatory.siblings.hydro.siteHint') },
        { value: 'cours_eau', label: t('observatory.siblings.hydro.coursEau'), hint: t('observatory.siblings.hydro.coursEauHint') },
      ]
  const activeLevel = isPiezo ? piezoLevel : hydroLevel
  const activeHint = levels.find(l => l.value === activeLevel)?.hint ?? ''

  // Expose isLoading and compute isDrawer before any early return (hooks already ran)
  const isLoading = isPiezo ? piezo.isLoading : hydro.isLoading
  const isDrawer = variant === 'drawer'

  if (isLoading) {
    if (isDrawer) return null
    return (
      <section className="bg-gray-900/50 rounded-xl border border-white/5 p-4">
        <div className="h-16 animate-pulse bg-white/5 rounded-lg" />
      </section>
    )
  }

  const data = isPiezo ? piezo.data : hydro.data
  if (!data) return null

  const nonRattachee = isPiezo && (piezo.data?.non_rattachee ?? false)
  const stationsWord = t('observatory.stations').toLowerCase()

  // "Infos communes" — group summary line (count + shared entity / watercourse + site)
  const commonInfo = isPiezo
    ? piezo.data?.code_bdlisa
      ? (
          <>
            {data.nb_stations} {stationsWord} · {t('observatory.siblings.entity')}{' '}
            <a
              href={`https://bdlisa.eaufrance.fr/hydrogeounit/${piezo.data.code_bdlisa}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:underline font-mono"
            >
              {piezo.data.code_bdlisa}
            </a>
          </>
        )
      : null
    : (
        <>
          {data.nb_stations} {stationsWord}
          {hydro.data?.nom_cours_eau ? ` · ${hydro.data.nom_cours_eau}` : ''}
          {hydro.data?.libelle_site ? ` · ${hydro.data.libelle_site}` : ''}
        </>
      )

  const rows: Row[] = isPiezo
    ? (piezo.data?.siblings ?? []).map(s => ({
        to: `/station/piezo/${encodeURIComponent(s.code_bss)}`,
        title: s.nom_commune || s.code_bss,
        subtitle: s.code_bss,
        classification: s.classification,
      }))
    : (hydro.data?.siblings ?? []).map(s => ({
        to: `/station/hydro/${s.code_station}`,
        title: s.libelle_station || s.code_station,
        subtitle: s.code_station,
        classification: s.classification,
      }))

  const hasClassified = rows.some(r => r.classification)

  const Toggle = !nonRattachee && (
    <div className="flex gap-1">
      {levels.map(l => (
        <button
          key={l.value}
          type="button"
          title={l.hint}
          aria-pressed={activeLevel === l.value}
          onClick={() => isPiezo
            ? setPiezoLevel(l.value as 'nappe' | 'systeme')
            : setHydroLevel(l.value as 'site' | 'cours_eau')}
          className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
            activeLevel === l.value
              ? 'bg-accent-cyan/20 text-accent-cyan'
              : 'text-text-secondary hover:bg-bg-hover'
          }`}
        >
          {l.label}
        </button>
      ))}
    </div>
  )

  const header = (
    <div className="flex items-center justify-between gap-2 mb-1">
      <span className="flex items-center gap-1.5 text-sm font-semibold text-gray-300">
        <Waves className="w-4 h-4" />
        {t('observatory.siblings.title')}
        <span title={t('observatory.siblings.help')} aria-label={t('observatory.siblings.help')} className="inline-flex cursor-help">
          <Info className="w-3 h-3 text-gray-500" />
        </span>
      </span>
      {Toggle}
    </div>
  )

  // Visible caption for the active level (more discoverable than a hover tooltip)
  const caption = !nonRattachee && activeHint && (
    <p className={`${isDrawer ? 'text-[10px] text-text-secondary' : 'text-[11px] text-gray-500'} mb-1`}>
      {activeHint}
    </p>
  )

  const common = !nonRattachee && commonInfo && (
    <p className={`${isDrawer ? 'text-xs text-text-secondary' : 'text-xs text-gray-400'} mb-2`}>{commonInfo}</p>
  )

  const legend = !isDrawer && !nonRattachee && hasClassified && (
    <div className="flex flex-wrap gap-x-2 gap-y-0.5 mt-2 pt-2 border-t border-white/5">
      {CLASSIFICATION_ORDER.map(c => (
        <span key={c} className="flex items-center gap-1 text-[9px] text-gray-500">
          <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: CLASSIFICATION_COLORS[c] }} />
          {CLASSIFICATION_LABELS[c]}
        </span>
      ))}
    </div>
  )

  const body = nonRattachee ? (
    <p className="text-xs text-text-secondary">{t('observatory.siblings.notLinked')}</p>
  ) : rows.length === 0 ? (
    <p className="text-xs text-text-secondary">{t('observatory.siblings.empty')}</p>
  ) : (
    <div className={`space-y-1 overflow-y-auto ${isDrawer ? 'max-h-48' : 'max-h-64'}`}>
      {(isDrawer ? rows.slice(0, 5) : rows).map(r => (
        <Link
          key={r.to}
          to={r.to}
          className="flex items-center justify-between py-1.5 px-2 rounded-lg hover:bg-bg-hover transition-colors"
        >
          <span className="text-xs text-gray-200 truncate">{r.title}</span>
          <span className="flex items-center gap-2 shrink-0 ml-2">
            {!isDrawer && r.subtitle && (
              <span className="text-[10px] text-gray-500">{r.subtitle}</span>
            )}
            {r.classification && (
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: CLASSIFICATION_COLORS[r.classification] ?? '#6b7280' }}
                title={CLASSIFICATION_LABELS[r.classification] ?? r.classification}
              />
            )}
          </span>
        </Link>
      ))}
    </div>
  )

  if (isDrawer) {
    return (
      <div className="bg-white/[0.03] rounded-lg p-3 border border-white/5">
        {header}
        {caption}
        {common}
        {body}
      </div>
    )
  }

  return (
    <section className="bg-gray-900/50 rounded-xl border border-white/5 p-4">
      {header}
      {caption}
      {common}
      {body}
      {legend}
    </section>
  )
}
