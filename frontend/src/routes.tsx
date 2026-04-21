import { lazy, Suspense } from 'react'
import { createBrowserRouter, Navigate } from 'react-router-dom'
import { Layout } from './components/layout/Layout'

const ObservatoryPage = lazy(() => import('./pages/ObservatoryPage'))
const StationPage = lazy(() => import('./pages/StationPage'))
const DashboardPage = lazy(() => import('./pages/DashboardPage'))

const PastasLayout = lazy(() => import('./pages/pastas/PastasLayout'))
const StationStep = lazy(() => import('./pages/pastas/StationStep'))
const CalibrateStep = lazy(() => import('./pages/pastas/CalibrateStep'))
const ResultsStep = lazy(() => import('./pages/pastas/ResultsStep'))
const ScenariosStep = lazy(() => import('./pages/pastas/ScenariosStep'))
const PastasGalleryPage = lazy(() => import('./pages/pastas/GalleryPage'))

const AILabLayout = lazy(() => import('./pages/ai/AILabLayout'))
const DataPage = lazy(() => import('./pages/DataPage'))
const TrainingPage = lazy(() => import('./pages/TrainingPage'))
const ForecastingPage = lazy(() => import('./pages/ForecastingPage'))

function SW({ children }: { children: React.ReactNode }) {
  return <Suspense fallback={<div className="flex items-center justify-center h-full text-text-secondary">Loading...</div>}>{children}</Suspense>
}

export const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      // Observatory (home)
      { path: '/', element: <SW><ObservatoryPage /></SW> },
      { path: '/observatory', element: <Navigate to="/" replace /> },
      { path: '/station/*', element: <SW><StationPage /></SW> },

      // Pastas Lab
      {
        path: '/pastas',
        element: <SW><PastasLayout /></SW>,
        children: [
          { index: true, element: <Navigate to="/pastas/station" replace /> },
          { path: 'station', element: <SW><StationStep /></SW> },
          { path: 'calibrate', element: <SW><CalibrateStep /></SW> },
          { path: 'results', element: <SW><ResultsStep /></SW> },
          { path: 'scenarios', element: <SW><ScenariosStep /></SW> },
          { path: 'gallery', element: <SW><PastasGalleryPage /></SW> },
          // Backward compatibility
          { path: 'fit', element: <Navigate to="/pastas/station" replace /> },
          { path: 'compare', element: <Navigate to="/pastas/gallery" replace /> },
        ],
      },

      // AI Lab
      {
        path: '/ai',
        element: <SW><AILabLayout /></SW>,
        children: [
          { index: true, element: <Navigate to="/ai/data" replace /> },
          { path: 'data', element: <SW><DataPage /></SW> },
          { path: 'training', element: <SW><TrainingPage /></SW> },
          { path: 'forecasting', element: <SW><ForecastingPage /></SW> },
        ],
      },

      // Backward compat redirects
      { path: '/data', element: <Navigate to="/ai/data" replace /> },
      { path: '/training', element: <Navigate to="/ai/training" replace /> },
      { path: '/forecasting', element: <Navigate to="/ai/forecasting" replace /> },
      { path: '/dashboard', element: <SW><DashboardPage /></SW> },

      // 404
      { path: '*', element: <div className="flex flex-col items-center justify-center h-full gap-4"><h1 className="text-4xl font-bold text-text-primary">404</h1><p className="text-text-secondary">Page not found</p></div> },
    ],
  },
])
