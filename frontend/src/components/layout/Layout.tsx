import { Outlet } from 'react-router-dom'
import { useTranslation } from 'react-i18next'
import { TopNav } from './TopNav'
import { OnboardingTour } from '../OnboardingTour'
import { CompareSelectionProvider } from '@/contexts/CompareSelection'

export function Layout() {
  const { t } = useTranslation()
  return (
    <CompareSelectionProvider>
      <div className="flex flex-col h-screen bg-bg-primary text-text-primary font-sans">
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:bg-accent-cyan focus:text-white focus:p-2 focus:rounded focus:z-[100]"
        >
          {t('common.back')}
        </a>

        <TopNav />

        <main id="main-content" className="flex-1 overflow-auto">
          <Outlet />
        </main>

        <OnboardingTour />
      </div>
    </CompareSelectionProvider>
  )
}
