import { useState } from 'react'
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
  thresholdValues?: (number | null)[] | null  // real-unit values at the 6 class boundaries (same month)
  measureUnit: string
  referenceFlag?: string | null    // 'normale' | 'adaptee' | 'provisoire' | null
  indexClassBounds?: number[] | null  // 6 ascending boundary values in physical units
}

function InfoDot({ tip }: { tip: string }) {
  return (
    <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-white/10 text-[9px] text-text-secondary cursor-help align-middle" title={tip} aria-label={tip}>i</span>
  )
}

export function SituationPanel(props: Props) {
  const { t, i18n } = useTranslation()
  const [rangesOpen, setRangesOpen] = useState(false)
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
          <span className="text-[10px] text-text-secondary">{props.indexName}
            {props.indexValue != null && <span className="ml-1 font-mono text-text-primary">{props.indexValue > 0 ? '+' : ''}{props.indexValue.toFixed(2).replace('.', ',')}</span>} <InfoDot tip={indexTip} />
          </span>
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
            const dec = isPiezo ? 2 : 1
            const tv = props.thresholdValues
            const hasReal = Array.isArray(tv) && tv.length === 6
            const fmtZ = (z: number) => `${z > 0 ? '+' : ''}${z.toFixed(2).replace('.', ',')}`
            const boundaryLabel = (j: number) =>
              hasReal ? (tv![j - 1] != null ? formatNumber(tv![j - 1] as number, dec) : '—') : fmtZ(ZB[j])
            return (
              <div className="mb-2">
                <div className="relative h-3.5">
                  {props.refValue != null && (
                    <span className="absolute -translate-x-1/2 text-[10px] font-mono font-semibold whitespace-nowrap" style={{ left: `${markerPct}%`, color }}>{formatNumber(props.refValue, dec)}</span>
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
                    <span key={j} className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: `${(j / 7) * 100}%` }}>{boundaryLabel(j)}</span>
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

      <ReferenceSection
        referenceFlag={props.referenceFlag}
        indexClassBounds={props.indexClassBounds}
        baselineStart={props.baselineStart}
        baselineEnd={props.baselineEnd}
        measureUnit={props.measureUnit}
        rangesOpen={rangesOpen}
        setRangesOpen={setRangesOpen}
        t={t}
      />
    </div>
  )
}

interface ReferenceSectionProps {
  referenceFlag?: string | null
  indexClassBounds?: number[] | null
  baselineStart?: string | null
  baselineEnd?: string | null
  measureUnit: string
  rangesOpen: boolean
  setRangesOpen: (v: boolean) => void
  t: (key: string, opts?: Record<string, string | number>) => string
}

function ReferenceSection({ referenceFlag, indexClassBounds, baselineStart, baselineEnd, measureUnit, rangesOpen, setRangesOpen, t }: ReferenceSectionProps) {
  const hasBounds = Array.isArray(indexClassBounds) && indexClassBounds.length === 6
  const hasFlag = referenceFlag != null

  if (!hasBounds && !hasFlag) return null

  // Build reference period caption
  let refCaption: string | null = null
  if (referenceFlag === 'normale') {
    refCaption = t('observatory.reference.normale')
  } else if (referenceFlag === 'adaptee' && baselineStart && baselineEnd) {
    refCaption = t('observatory.reference.adaptee', { start: baselineStart.slice(0, 4), end: baselineEnd.slice(0, 4) })
  } else if (referenceFlag === 'provisoire') {
    refCaption = t('observatory.reference.provisoire')
  } else if (!hasBounds) {
    refCaption = t('observatory.reference.unavailable')
  }

  return (
    <div className="mt-2 border-t border-white/5 pt-2">
      <button
        type="button"
        className="flex items-center gap-1 w-full text-left"
        onClick={() => setRangesOpen(!rangesOpen)}
        aria-expanded={rangesOpen}
      >
        <span className="text-[10px] uppercase tracking-wider text-text-secondary flex-1">
          {t('observatory.reference.title')} ({measureUnit})
        </span>
        <span className="text-[10px] text-text-secondary select-none">{rangesOpen ? '▲' : '▼'}</span>
      </button>

      {refCaption && (
        <div className="text-[10px] text-text-secondary mt-0.5 italic">{refCaption}</div>
      )}

      {rangesOpen && hasBounds && (
        <div className="mt-1.5 space-y-0.5">
          {CLASSIFICATION_ORDER.map((cls, idx) => {
            const b = indexClassBounds!
            const dec = measureUnit.includes('m³') ? 1 : 2
            const lo = idx === 0 ? null : b[idx - 1]
            const hi = idx === 6 ? null : b[idx]
            const loStr = lo == null ? '−∞' : formatNumber(lo, dec)
            const hiStr = hi == null ? '+∞' : formatNumber(hi, dec)
            const color = CLASSIFICATION_COLORS[cls as string] ?? '#6b7280'
            return (
              <div key={cls} className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-sm flex-shrink-0" style={{ backgroundColor: color }} />
                <span className="text-[10px] text-text-secondary flex-1 leading-tight">
                  {CLASSIFICATION_LABELS[cls as string] ?? cls}
                </span>
                <span className="text-[10px] font-mono text-text-primary whitespace-nowrap">
                  {loStr} – {hiStr}
                </span>
              </div>
            )
          })}
        </div>
      )}

      {rangesOpen && !hasBounds && (
        <div className="mt-1 text-[10px] text-text-secondary italic">{t('observatory.reference.unavailable')}</div>
      )}
    </div>
  )
}
