import { lazy, Suspense } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import { Layout } from './components/layout/Layout'

const DashboardPage = lazy(() => import('./pages/DashboardPage'))
const DataPage = lazy(() => import('./pages/DataPage'))
const TrainingPage = lazy(() => import('./pages/TrainingPage'))
const ForecastingPage = lazy(() => import('./pages/ForecastingPage'))
const ObservatoryPage = lazy(() => import('./pages/ObservatoryPage'))
const PastasLayout = lazy(() => import('./pages/pastas/PastasLayout'))
const PastasFitPage = lazy(() => import('./pages/pastas/FitPage'))
const PastasScenariosPage = lazy(() => import('./pages/pastas/ScenariosPage'))
const PastasGalleryPage = lazy(() => import('./pages/pastas/GalleryPage'))
const PastasComparePage = lazy(() => import('./pages/pastas/ComparePage'))

function SuspenseWrapper({ children }: { children: React.ReactNode }) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-full text-text-secondary">
          Loading...
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
      <p className="text-text-secondary">Page not found</p>
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
        path: '/observatory',
        element: (
          <SuspenseWrapper>
            <ObservatoryPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/pastas',
        element: (
          <SuspenseWrapper>
            <PastasLayout />
          </SuspenseWrapper>
        ),
        children: [
          {
            index: true,
            element: <Navigate to="/pastas/fit" replace />,
          },
          {
            path: 'fit',
            element: (
              <SuspenseWrapper>
                <PastasFitPage />
              </SuspenseWrapper>
            ),
          },
          {
            path: 'scenarios',
            element: (
              <SuspenseWrapper>
                <PastasScenariosPage />
              </SuspenseWrapper>
            ),
          },
          {
            path: 'gallery',
            element: (
              <SuspenseWrapper>
                <PastasGalleryPage />
              </SuspenseWrapper>
            ),
          },
          {
            path: 'compare',
            element: (
              <SuspenseWrapper>
                <PastasComparePage />
              </SuspenseWrapper>
            ),
          },
        ],
      },
      {
        path: '/lab/*',
        element: <Navigate to="/" replace />,
      },
      {
        path: '*',
        element: <NotFound />,
      },
    ],
  },
])
