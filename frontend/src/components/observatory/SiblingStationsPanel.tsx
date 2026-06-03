import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Waves } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { usePiezoSiblings, useHydroSiblings } from '@/hooks/useObservatory'
import { CLASSIFICATION_COLORS } from '@/lib/observatory-constants'

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

  // Fix 1: expose isLoading and compute isDrawer before any early return
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

  // Build a uniform shape from the two payloads
  const data = isPiezo ? piezo.data : hydro.data
  if (!data) return null

  const nonRattachee = isPiezo && (piezo.data?.non_rattachee ?? false)
  const subtitle = isPiezo
    ? piezo.data?.code_bdlisa ?? ''
    : `${hydro.data?.libelle_site || hydro.data?.code_site || ''}${hydro.data?.nom_cours_eau ? ` - ${hydro.data.nom_cours_eau}` : ''}`

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

  // Fix 2: call typed setters directly; Fix 3: type="button" + aria-pressed
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
      <span className="flex items-center gap-2 text-sm font-semibold text-gray-300">
        <Waves className="w-4 h-4" />
        {t('observatory.siblings.title')}
      </span>
      {Toggle}
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
                title={r.classification}
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
        {subtitle && !nonRattachee && (
          <p className="text-xs text-text-secondary mb-2">{subtitle}</p>
        )}
        {body}
      </div>
    )
  }

  return (
    <section className="bg-gray-900/50 rounded-xl border border-white/5 p-4">
      {header}
      {subtitle && !nonRattachee && <p className="text-xs text-gray-500 mb-3">{subtitle}</p>}
      {body}
    </section>
  )
}
