import { useTranslation } from 'react-i18next'
import { CLIMAT_VARIABLES, climatGradientCss, climatRawDomain, isClimatIndexVariable, isClimatDailyVariable, PRECIP_DAILY_BOUNDS, PRECIP_DAILY_COLORS } from '@/lib/climat-colors'
import type { ClimatVariable } from '@/lib/climat-colors'
import { SPI_CLASS_COLORS, SPI_CLASS_ORDER, STI_CLASS_COLORS, STI_CLASS_ORDER } from '@/lib/era5-colors'
import { formatDayLabel } from '@/lib/climat-day-stepper'

interface Props {
  variable: ClimatVariable
  window: number
  /** Active month ('YYYY-MM'), or for daily-temp variables the active day ('YYYY-MM-DD'). */
  month: string
  /** True when the raw-variable grid-monthly response came back with
   *  mois_complet=false for this month (partial current month) — shows a
   *  "mois incomplet" badge so the value isn't read as a settled monthly figure. */
  incomplete?: boolean
}

/** 7-class McKee legend (SPI/STI) or gradient legend (raw/daily variables) for the Climat map. */
export function ClimatLegend({ variable, window, month, incomplete }: Props) {
  const { t, i18n } = useTranslation()
  const cfg = CLIMAT_VARIABLES[variable]
  const isDaily = isClimatDailyVariable(variable)

  const m = month.match(/^(\d{4})-(\d{2})/)
  const periodLabel = isDaily
    ? formatDayLabel(month, i18n.language)
    : m
      ? new Intl.DateTimeFormat(i18n.language, { month: 'short', year: 'numeric' }).format(new Date(Number(m[1]), Number(m[2]) - 1, 1))
      : month

  if (isClimatIndexVariable(variable)) {
    const isSpi = variable === 'spi'
    const order = isSpi ? SPI_CLASS_ORDER : STI_CLASS_ORDER
    const colors = isSpi ? SPI_CLASS_COLORS : STI_CLASS_COLORS
    const ns = isSpi ? 'observatory.spi' : 'observatory.sti'
    return (
      <div className="absolute bottom-4 left-3 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-3 py-2 shadow-lg pointer-events-none" style={{ maxWidth: '190px' }}>
        <div className="text-xs font-semibold text-text-primary leading-tight">{t(cfg.labelKey)}</div>
        <div className="text-[10px] text-text-secondary mt-0.5">
          {t('climat.legend.window', { n: window })} · {periodLabel}
        </div>
        <div className="mt-1.5 space-y-0.5">
          {[...order].reverse().map((cls) => (
            <div key={cls} className="flex items-center gap-1.5">
              <span className="w-3 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: colors[cls] }} />
              <span className="text-[9px] text-text-secondary">{t(`${ns}.${cls}`, { defaultValue: cls })}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (variable === 'bilan_hydrique') {
    return (
      <div className="absolute bottom-4 left-3 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-3 py-2 shadow-lg pointer-events-none" style={{ maxWidth: '190px' }}>
        <div className="text-xs font-semibold text-text-primary leading-tight">{t(cfg.labelKey)}</div>
        <div className="text-[10px] text-text-secondary mt-0.5">{periodLabel}</div>
        {incomplete && (
          <div className="text-[9px] font-semibold text-amber-400 mt-0.5">{t('climat.legend.incompleteMonth')}</div>
        )}
        <div className="mt-1.5 space-y-0.5">
          {[...SPI_CLASS_ORDER].reverse().map((cls) => (
            <div key={cls} className="flex items-center gap-1.5">
              <span className="w-3 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: SPI_CLASS_COLORS[cls] }} />
              <span className="text-[9px] text-text-secondary">{t(`climat.bilanClasses.${cls}`, { defaultValue: cls })}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (variable === 'precip_daily') {
    // Bornes en mm, sans noms de classes : contrairement au SPI (où « Très sec »
    // porte l'information, le z-score étant illisible), ici la valeur EST le sens.
    return (
      <div className="absolute bottom-4 left-3 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-3 py-2 shadow-lg pointer-events-none" style={{ maxWidth: '190px' }}>
        <div className="text-xs font-semibold text-text-primary leading-tight">{t(cfg.labelKey)}</div>
        <div className="text-[10px] text-text-secondary mt-0.5">{periodLabel}</div>
        <div className="mt-1.5 space-y-0.5">
          {[...PRECIP_DAILY_COLORS].map((color, i) => {
            const label = i === 0
              ? t('climat.legend.precipDryClass')
              : i === PRECIP_DAILY_COLORS.length - 1
                ? `≥ ${PRECIP_DAILY_BOUNDS[i - 1]}`
                : `${PRECIP_DAILY_BOUNDS[i - 1]} – ${PRECIP_DAILY_BOUNDS[i]}`
            return (
              <div key={color} className="flex items-center gap-1.5">
                <span className="w-3 h-2.5 rounded-sm flex-shrink-0" style={{ backgroundColor: color }} />
                <span className="text-[9px] text-text-secondary">{label} {cfg.unit}</span>
              </div>
            )
          }).reverse()}
        </div>
      </div>
    )
  }

  const [minVal, maxVal] = climatRawDomain(variable)
  return (
    <div className="absolute bottom-4 left-3 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-3 py-2 shadow-lg pointer-events-none" style={{ maxWidth: '190px' }}>
      <div className="text-xs font-semibold text-text-primary leading-tight">{t(cfg.labelKey)}</div>
      <div className="text-[10px] text-text-secondary mt-0.5">{periodLabel}</div>
      {incomplete && (
        <div className="text-[9px] font-semibold text-amber-400 mt-0.5">{t('climat.legend.incompleteMonth')}</div>
      )}
      <div className="mt-1.5">
        <div className="h-2.5 rounded" style={{ background: climatGradientCss(variable) }} />
        <div className="relative flex justify-between text-[9px] text-text-secondary mt-0.5">
          <span>{String(minVal).replace('-', '−')} {cfg.unit}</span>
          <span>+{maxVal} {cfg.unit}</span>
        </div>
      </div>
    </div>
  )
}
