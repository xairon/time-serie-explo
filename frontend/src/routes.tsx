import { lazy, Suspense } from 'react'
import { createBrowserRouter } from 'react-router-dom'
import { Layout } from './components/layout/Layout'

const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const DataPage = lazy(() => import('./pages/DataPage'))
const TrainingPage = lazy(() => import('./pages/TrainingPage'))
const ForecastingPage = lazy(() => import('./pages/ForecastingPage'))
const CounterfactualPage = lazy(() => import('./pages/CounterfactualPage'))
const ObservatoryPage = lazy(() => import('./pages/ObservatoryPage'))
const PumpingDetectionPage = lazy(() => import('./pages/PumpingDetectionPage'))

function SuspenseWrapper({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-full text-text-secondary">
          Chargement...
        </div>
      }
    >
      {children}
    </Suspense>
  )
}

function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4">
      <h1 className="text-4xl font-bold text-text-primary">404</h1>
      <p className="text-text-secondary">Page non trouvée</p>
    </div>
  )
}

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      {
        path: '/',
        element: (
          <SuspenseWrapper>
            <DashboardPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/data',
        element: (
          <SuspenseWrapper>
            <DataPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/training',
        element: (
          <SuspenseWrapper>
            <TrainingPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/forecasting',
        element: (
          <SuspenseWrapper>
            <ForecastingPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/counterfactual',
        element: (
          <SuspenseWrapper>
            <CounterfactualPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/observatory',
        element: (
          <SuspenseWrapper>
            <ObservatoryPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/pumping-detection',
        element: (
          <SuspenseWrapper>
            <PumpingDetectionPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '*',
        element: <NotFound />,
      },
    ],
  },
])
