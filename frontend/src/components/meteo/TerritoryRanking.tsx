import { useTranslation } from 'react-i18next'
import type { TerritorySituation } from '@/lib/observatory-types'
import { classColor, trendGlyph, CLASS_ORDER_DRIEST_FIRST } from '@/lib/situation-format'

export function sortByCriticality(ts: TerritorySituation[]): TerritorySituation[] {
  const rank = (t: TerritorySituation) =>
    t.situation_class ? CLASS_ORDER_DRIEST_FIRST.indexOf(t.situation_class) : 999
  return [...ts].sort((a, b) => rank(a) - rank(b))
}

export function TerritoryRanking({ territories, onSelect }: { territories: TerritorySituation[]; onSelect: (code: string) => void }) {
  const { t } = useTranslation()
  const rows = sortByCriticality(territories)
  return (
    <div className="bg-bg-card border border-white/5 rounded-xl p-4">
      <h2 className="text-sm font-semibold text-text-primary mb-3">{t('meteo.rankingTitle')}</h2>
      <ul className="space-y-1">
        {rows.map(tr => (
          <li key={tr.code}>
            <button onClick={() => onSelect(tr.code)}
              className="w-full flex items-center gap-3 px-2 py-1.5 rounded hover:bg-bg-hover text-left">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: classColor(tr.situation_class) }} />
              <span className="flex-1 text-sm text-text-primary truncate">{tr.name}</span>
              <span className="text-xs text-text-secondary">{tr.insufficient ? t('meteo.insufficient') : `${tr.pct_below_normal ?? 0}%`}</span>
              <span aria-hidden style={{ color: classColor(tr.situation_class) }}>{trendGlyph(tr.trend)}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
