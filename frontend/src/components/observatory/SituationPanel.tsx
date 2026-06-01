import { useTranslation } from 'react-i18next'
import { CLASSIFICATION_COLORS, CLASSIFICATION_LABELS, CLASSIFICATION_ORDER } from '@/lib/observatory-constants'
import { formatNumber } from '@/lib/observatory-utils'

interface Props {
  type: 'piezo' | 'hydro'
  indexName?: string | null
  indexValue?: number | null
  indexClass?: string | null
  refMonth?: string | null
  baselineStart?: string | null
  baselineEnd?: string | null
  refValue?: number | null      // value of the reference month (the value actually classified)
  monthMedian?: number | null   // median of that same calendar month across years
  measureUnit: string
}

function InfoDot({ tip }: { tip: string }) {
  return (
    <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-white/10 text-[9px] text-text-secondary cursor-help align-middle" title={tip} aria-label={tip}>i</span>
  )
}

export function SituationPanel(props: Props) {
  const { t, i18n } = useTranslation()
  const isPiezo = props.type === 'piezo'
  const cls = props.indexClass
  const unknown = !cls || cls === 'UNKNOWN'
  const color = (cls && CLASSIFICATION_COLORS[cls]) || '#6b7280'
  const indexTip = isPiezo ? t('observatory.situation.ipsTip') : t('observatory.situation.ssfiTip')

  return (
    <div className="bg-white/[0.03] rounded-lg p-3 border border-white/5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] uppercase tracking-wider text-text-secondary">
          {isPiezo ? t('observatory.situation.title') : t('observatory.situation.titleHydro')}
        </span>
        {props.indexName && (
          <span className="text-[10px] text-text-secondary">{props.indexName} <InfoDot tip={indexTip} /></span>
        )}
      </div>

      {unknown ? (
        <div className="text-xs text-text-secondary">{t('observatory.situation.unclassified')}</div>
      ) : (
        <>
          <div className="flex items-center gap-1.5 mb-2">
            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-sm font-semibold" style={{ color }}>{CLASSIFICATION_LABELS[cls!] ?? cls}</span>
          </div>
          {(() => {
            const ZB = [-2.5, -1.75, -1.28, -0.84, 0.84, 1.28, 1.75, 2.5]
            const idx = CLASSIFICATION_ORDER.indexOf(cls as any)
            const v = props.indexValue
            let frac = 0.5
            if (idx >= 0 && v != null) {
              const lo = ZB[idx], hi = ZB[idx + 1]
              frac = Math.max(0, Math.min(1, (v - lo) / (hi - lo)))
            }
            const markerPct = idx >= 0 ? ((idx + frac) / 7) * 100 : 50
            const fmt = (z: number) => `${z > 0 ? '+' : ''}${z.toFixed(2).replace('.', ',')}`
            return (
              <div className="mb-2">
                <div className="relative h-3.5">
                  {v != null && (
                    <span className="absolute -translate-x-1/2 text-[10px] font-mono font-semibold whitespace-nowrap" style={{ left: `${markerPct}%`, color }}>{fmt(v)}</span>
                  )}
                </div>
                <div className="relative" role="img" aria-label={CLASSIFICATION_LABELS[cls!] ?? cls!}>
                  <div className="flex gap-0.5 h-2.5">
                    {CLASSIFICATION_ORDER.map(c => (
                      <span key={c} className="flex-1 rounded-sm" style={{
                        backgroundColor: CLASSIFICATION_COLORS[c as string],
                        opacity: c === cls ? 1 : 0.4,
                      }} />
                    ))}
                  </div>
                  {v != null && (
                    <span className="absolute -top-0.5 -bottom-0.5 w-0.5 bg-white rounded-full" style={{ left: `${markerPct}%`, transform: 'translateX(-50%)' }} />
                  )}
                </div>
                <div className="relative h-3 mt-0.5 text-[8px] text-text-secondary font-mono">
                  {[1, 2, 3, 4, 5, 6].map(j => (
                    <span key={j} className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: `${(j / 7) * 100}%` }}>{fmt(ZB[j])}</span>
                  ))}
                </div>
                <div className="flex justify-between text-[9px] text-text-secondary">
                  <span>{t('observatory.situation.scaleLow')}</span><span>{t('observatory.situation.scaleHigh')}</span>
                </div>
              </div>
            )
          })()}
        </>
      )}

      {props.refMonth && (() => {
        const locale = i18n.language?.startsWith('en') ? 'en-GB' : 'fr-FR'
        const d = new Date(props.refMonth!)
        const monthYear = d.toLocaleDateString(locale, { month: 'long', year: 'numeric' })
        const monthName = d.toLocaleDateString(locale, { month: 'long' })
        const dec = isPiezo ? 2 : 1
        return (
          <div className="space-y-0.5">
            {props.refValue != null && (
              <div className="text-xs">
                <span className="text-text-secondary capitalize">{monthYear}</span>
                <span className="text-text-secondary"> : </span>
                <span className="text-text-primary font-mono">{formatNumber(props.refValue, dec)} {props.measureUnit}</span>
                {isPiezo && <> <InfoDot tip={t('observatory.situation.ngfTip')} /></>}
              </div>
            )}
            {props.monthMedian != null && (
              <div className="text-[11px] text-text-secondary">
                {t('observatory.situation.typicalForMonth', { month: monthName })} : <span className="font-mono">{formatNumber(props.monthMedian, dec)} {props.measureUnit}</span>
              </div>
            )}
            {props.baselineStart && props.baselineEnd && (
              <div className="text-[10px] text-text-secondary">{props.baselineStart.slice(0, 4)}–{props.baselineEnd.slice(0, 4)}</div>
            )}
          </div>
        )
      })()}
    </div>
  )
}
