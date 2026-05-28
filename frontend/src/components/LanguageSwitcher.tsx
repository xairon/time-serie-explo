import { useTranslation } from 'react-i18next'
import { Languages } from 'lucide-react'

export function LanguageSwitcher() {
  const { i18n, t } = useTranslation()
  const current = i18n.language.startsWith('en') ? 'en' : 'fr'
  const other = current === 'fr' ? 'en' : 'fr'

  const change = (lng: 'fr' | 'en') => {
    void i18n.changeLanguage(lng)
  }

  return (
    <button
      onClick={() => change(other)}
      title={t('language.switchTo', { lang: other === 'fr' ? 'Français' : 'English' })}
      aria-label={t('language.switchTo', { lang: other === 'fr' ? 'Français' : 'English' })}
      className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs text-text-secondary hover:bg-bg-hover hover:text-text-primary transition-colors"
    >
      <Languages className="w-3.5 h-3.5" />
      <span className="font-mono uppercase">{current}</span>
    </button>
  )
}
