import { useTranslation } from 'react-i18next'
import { Sparkles } from 'lucide-react'
import type { Outlook } from '@/lib/observatory-types'

export function OutlookPanel({ outlook }: { outlook: Outlook | null }) {
  const { t } = useTranslation()
  return (
    <div className="bg-bg-card border border-dashed border-accent-indigo/40 rounded-xl p-4 flex items-center gap-3 opacity-80">
      <Sparkles className="w-4 h-4 text-accent-indigo" />
      <div>
        <div className="text-sm font-medium text-text-primary">{t('meteo.outlookTitle')}</div>
        <div className="text-xs text-text-secondary">
          {outlook ? `${outlook.horizon_months} mois` : t('meteo.outlookSoon')}
        </div>
      </div>
    </div>
  )
}
