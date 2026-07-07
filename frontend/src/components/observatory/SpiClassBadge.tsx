import { useTranslation } from 'react-i18next'
import { spiClassBadge } from '@/lib/climate-cumuls'

/** Small pill showing a WMO/McKee SPI class with its BrBG colour (era5-colors). */
export function SpiClassBadge({ cls, className = '' }: { cls: string | null | undefined; className?: string }) {
  const { t } = useTranslation()
  const badge = spiClassBadge(cls)
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold ${className}`}
      style={{ backgroundColor: badge.color, color: badge.textColor }}
    >
      {t(badge.labelKey)}
    </span>
  )
}
