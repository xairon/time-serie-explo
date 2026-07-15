import { useTranslation } from 'react-i18next'
import type { ClimatSituationSummary } from '@/lib/observatory-types'
import { buildSituationBannerData } from '@/lib/climat-situation-format'
import { SPI_CLASS_ORDER, SPI_CLASS_COLORS } from '@/lib/era5-colors'

interface Props {
  summary: ClimatSituationSummary | undefined
  isLoading: boolean
}

/** Territory-wide synthesis banner: "X % du territoire en sécheresse (SPI < −1) ·
 *  mois le plus sec depuis AAAA" + a stacked bar showing the 7-class SPI
 *  distribution (`summary.classes_pct`, driest → wettest via SPI_CLASS_ORDER) —
 *  fed by GET /observatory/climat/situation-summary. No raw lat/lon is
 *  rendered anymore (replaces the old "zones les plus touchées" chips).
 *  Text shaping lives in climat-situation-format.ts (unit-tested
 *  independently of this component). */
export function SituationBanner({ summary, isLoading }: Props) {
  const { t } = useTranslation()

  if (isLoading || !summary) {
    return (
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-4 py-2 shadow-lg text-xs text-text-secondary">
        {t('climat.banner.loading')}
      </div>
    )
  }

  // No SPI computed for this month/window yet (e.g. the partial current month)
  // — never render this as a real "0 % en sécheresse".
  if (!summary.available) {
    return (
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-4 py-2 shadow-lg text-xs text-text-secondary">
        {t('climat.banner.indicesUnavailable')}
      </div>
    )
  }

  const data = buildSituationBannerData(summary)
  const classes = SPI_CLASS_ORDER.filter((c) => (summary.classes_pct[c] ?? 0) > 0)

  return (
    <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 max-w-2xl bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-4 py-2.5 shadow-lg">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-text-primary">
        <span className="font-semibold text-accent-cyan">{t('climat.banner.droughtPct', { pct: data.pctSecheresse })}</span>
        {data.driestSinceYear != null && (
          <>
            <span className="text-text-muted">·</span>
            <span>{t('climat.banner.driestSince', { year: data.driestSinceYear })}</span>
          </>
        )}
      </div>
      <div
        role="img"
        aria-label={t('climat.banner.distributionAria')}
        className="flex w-full h-2.5 mt-2 rounded-full overflow-hidden bg-bg-hover"
      >
        {classes.map((c) => (
          <span
            key={c}
            title={`${summary.classes_pct[c]} %`}
            style={{ width: `${summary.classes_pct[c]}%`, background: SPI_CLASS_COLORS[c] }}
          />
        ))}
      </div>
    </div>
  )
}
