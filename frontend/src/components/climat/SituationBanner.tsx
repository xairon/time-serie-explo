import { useTranslation } from 'react-i18next'
import type { ClimatSituationSummary } from '@/lib/observatory-types'
import { buildSituationBannerData } from '@/lib/climat-situation-format'

interface Props {
  summary: ClimatSituationSummary | undefined
  isLoading: boolean
}

/** Territory-wide synthesis banner: "X % du territoire en sécheresse (SPI < −1) ·
 *  mois le plus sec depuis AAAA · zones les plus touchées : …" — fed by
 *  GET /observatory/climat/situation-summary. Text shaping lives in
 *  climat-situation-format.ts (unit-tested independently of this component). */
export function SituationBanner({ summary, isLoading }: Props) {
  const { t } = useTranslation()

  if (isLoading || !summary) {
    return (
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-4 py-2 shadow-lg text-xs text-text-secondary">
        {t('climat.banner.loading')}
      </div>
    )
  }

  const data = buildSituationBannerData(summary)

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
      {data.chips.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
          <span className="text-[10px] text-text-secondary">{t('climat.banner.mostAffected')}</span>
          {data.chips.map((chip) => (
            <span
              key={chip.label}
              className="text-[10px] px-1.5 py-0.5 rounded-full bg-bg-hover text-text-secondary border border-white/10"
            >
              {chip.label}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
