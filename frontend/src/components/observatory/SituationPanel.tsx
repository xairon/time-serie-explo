import { useTranslation } from 'react-i18next'
import { CLASSIFICATION_COLORS, CLASSIFICATION_LABELS, CLASSIFICATION_ORDER } from '@/lib/observatory-constants'
import { formatNumber, formatDate } from '@/lib/observatory-utils'

interface Props {
  type: 'piezo' | 'hydro'
  indexName?: string | null
  indexValue?: number | null
  indexClass?: string | null
  refMonth?: string | null
  baselineStart?: string | null
  baselineEnd?: string | null
  measure?: number | null
  measureUnit: string
}

function InfoDot({ tip }: { tip: string }) {
  return (
    <span className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-white/10 text-[9px] text-text-secondary cursor-help align-middle" title={tip} aria-label={tip}>i</span>
  )
}

export function SituationPanel(props: Props) {
  const { t } = useTranslation()
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
          <span className="text-[10px] text-text-secondary">{props.indexName} <InfoDot tip={indexTip} />
            {props.indexValue != null && <span className="ml-1 font-mono text-text-primary">{props.indexValue.toFixed(2)}</span>}
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
          <div className="flex gap-0.5 mb-1" role="img" aria-label={CLASSIFICATION_LABELS[cls!] ?? cls!}>
            {CLASSIFICATION_ORDER.map(c => (
              <span key={c} className="h-2 flex-1 rounded-sm" style={{
                backgroundColor: CLASSIFICATION_COLORS[c as string],
                opacity: c === cls ? 1 : 0.25,
                outline: c === cls ? '1px solid rgba(255,255,255,0.8)' : 'none',
              }} />
            ))}
          </div>
          <div className="flex justify-between text-[9px] text-text-secondary mb-2">
            <span>{t('observatory.situation.scaleLow')}</span><span>{t('observatory.situation.scaleHigh')}</span>
          </div>
        </>
      )}

      {props.measure != null && (
        <div className="text-xs text-text-secondary">
          {t('observatory.situation.measure')} : <span className="text-text-primary font-mono">{formatNumber(props.measure, 2)} {props.measureUnit}</span>
          {isPiezo && <> <InfoDot tip={t('observatory.situation.ngfTip')} /></>}
        </div>
      )}
      {props.refMonth && <div className="text-[10px] text-text-secondary mt-1">{formatDate(props.refMonth)}{props.baselineStart && props.baselineEnd && <> · {props.baselineStart.slice(0,4)}–{props.baselineEnd.slice(0,4)}</>}</div>}
    </div>
  )
}
