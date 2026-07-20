import { useTranslation } from 'react-i18next'
import type { ClimatSituationSummary } from '@/lib/observatory-types'
import { SPI_CLASS_ORDER, SPI_CLASS_COLORS } from '@/lib/era5-colors'

interface Props {
  summary: ClimatSituationSummary | undefined
  isLoading: boolean
}

/** Barre de distribution 7 classes du territoire (`summary.classes_pct`, du plus
 *  sec au plus humide via SPI_CLASS_ORDER), alimentée par
 *  GET /observatory/climat/situation-summary. La part sèche s'y lit à l'œil
 *  (somme des 3 classes les plus sèches). La phrase de synthèse narrée a été
 *  retirée — elle faisait trop « généré » ; l'info survit dans la barre. */
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

  const classes = SPI_CLASS_ORDER.filter((c) => (summary.classes_pct[c] ?? 0) > 0)

  return (
    <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 max-w-2xl bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-4 py-2.5 shadow-lg">
      <div
        role="img"
        aria-label={t('climat.banner.distributionAria')}
        className="flex w-full h-2.5 rounded-full overflow-hidden bg-bg-hover"
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
