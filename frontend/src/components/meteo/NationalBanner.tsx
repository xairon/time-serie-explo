import { useTranslation } from 'react-i18next'
import type { NationalSituation } from '@/lib/observatory-types'
import { classColor, trendGlyph } from '@/lib/situation-format'

export function NationalBanner({ data, departmentsInAlert }: { data: NationalSituation; departmentsInAlert: number }) {
  const { t } = useTranslation()
  const color = classColor(data.situation_class)
  return (
    <div className="rounded-2xl border border-white/10 p-6 flex flex-wrap items-center gap-6"
         style={{ background: `linear-gradient(135deg, ${color}22, transparent)` }}>
      <div className="flex items-center gap-4">
        <span className="text-4xl" style={{ color }} aria-hidden>{trendGlyph(data.trend)}</span>
        <div>
          <div className="text-2xl font-bold" style={{ color }}>
            {data.insufficient || !data.situation_class
              ? t('meteo.insufficient')
              : t(`meteo.class.${data.situation_class}`)}
          </div>
          <div className="text-sm text-text-secondary">
            {t('meteo.trendGeneral')}: {data.trend ? t(`meteo.trend.${data.trend}`) : '—'}
          </div>
        </div>
      </div>
      {data.pct_below_normal != null && (
        <div className="text-text-primary text-lg font-mono">
          {t('meteo.belowNormal', { pct: data.pct_below_normal })}
        </div>
      )}
      <div className="text-text-secondary text-sm">
        {t('meteo.departmentsInAlert', { count: departmentsInAlert })}
      </div>
    </div>
  )
}
