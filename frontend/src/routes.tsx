import { lazy, Suspense } from 'react'
import { createBrowserRouter, Navigate, useSearchParams } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { RequireAuth } from './components/auth/RequireAuth'
import { SessionGate } from './components/auth/SessionGate'

function RedirectWithParams({ to }: { to: string }) {
  const [searchParams] = useSearchParams()
  const qs = searchParams.toString()
  return <Navigate to={qs ? `${to}?${qs}` : to} replace />
}

const ObservatoryPage = lazy(() => import('./pages/ObservatoryPage'))
const StationPage = lazy(() => import('./pages/StationPage'))
const ComparePage = lazy(() => import('./pages/ComparePage'))
const DashboardPage = lazy(() => import('./pages/DashboardPage'))

const PastasLayout = lazy(() => import('./pages/pastas/PastasLayout'))
const CalibrateStep = lazy(() => import('./pages/pastas/CalibrateStep'))
const ResultsStep = lazy(() => import('./pages/pastas/ResultsStep'))
const ScenariosStep = lazy(() => import('./pages/pastas/ScenariosStep'))
const PastasGalleryPage = lazy(() => import('./pages/pastas/GalleryPage'))

const AILabLayout = lazy(() => import('./pages/ai/AILabLayout'))
const DataPage = lazy(() => import('./pages/DataPage'))
const TrainingPage = lazy(() => import('./pages/TrainingPage'))
const ForecastingPage = lazy(() => import('./pages/ForecastingPage'))
const AIModelsPage = lazy(() => import('./pages/ai/AIModelsPage'))

const LoginPage = lazy(() => import('./pages/LoginPage'))
const PrivacyPage = lazy(() => import('./pages/PrivacyPage'))
const AccountPage = lazy(() => import('./pages/AccountPage'))
const UsersPage = lazy(() => import('./pages/admin/UsersPage'))

function SW({ children }: { children: React.ReactNode }) {
  return <Suspense fallback={<div className="flex items-center justify-center h-full text-text-secondary">Chargement...</div>}>{children}</Suspense>
}

export const router = createBrowserRouter([
  {
    element: <SessionGate><Layout /></SessionGate>,
    children: [
      // Observatory (home)
      { path: '/', element: <SW><ObservatoryPage /></SW> },
      { path: '/observatory', element: <Navigate to="/" replace /> },
      { path: '/meteo', element: <Navigate to="/" replace /> },
      { path: '/station/*', element: <SW><StationPage /></SW> },

      // Cross-station comparison
      { path: '/compare', element: <Navigate to="/compare/piezo" replace /> },
      { path: '/compare/piezo', element: <SW><ComparePage stationType="piezo" /></SW> },
      { path: '/compare/hydro', element: <SW><ComparePage stationType="hydro" /></SW> },

      // Pastas Lab
      {
        path: '/pastas',
        element: <RequireAuth><SW><PastasLayout /></SW></RequireAuth>,
        children: [
          { index: true, element: <Navigate to="/pastas/calibrate" replace /> },
          { path: 'station', element: <RedirectWithParams to="/pastas/calibrate" /> },
          { path: 'calibrate', element: <SW><CalibrateStep /></SW> },
          { path: 'results', element: <SW><ResultsStep /></SW> },
          { path: 'scenarios', element: <SW><ScenariosStep /></SW> },
          { path: 'gallery', element: <SW><PastasGalleryPage /></SW> },
          // Backward compatibility — preserve query params
          { path: 'fit', element: <RedirectWithParams to="/pastas/calibrate" /> },
          { path: 'compare', element: <RedirectWithParams to="/pastas/gallery" /> },
        ],
      },

      // AI Lab
      {
        path: '/ai',
        element: <RequireAuth><SW><AILabLayout /></SW></RequireAuth>,
        children: [
          { index: true, element: <Navigate to="/ai/data" replace /> },
          { path: 'data', element: <SW><DataPage /></SW> },
          { path: 'training', element: <SW><TrainingPage /></SW> },
          { path: 'forecasting', element: <SW><ForecastingPage /></SW> },
          { path: 'models', element: <SW><AIModelsPage /></SW> },
        ],
      },

      // Backward compat redirects
      { path: '/data', element: <RedirectWithParams to="/ai/data" /> },
      { path: '/training', element: <RedirectWithParams to="/ai/training" /> },
      { path: '/forecasting', element: <RedirectWithParams to="/ai/forecasting" /> },
      { path: '/dashboard', element: <SW><DashboardPage /></SW> },

      // Auth
      { path: '/login', element: <SW><LoginPage /></SW> },
      { path: '/privacy', element: <SW><PrivacyPage /></SW> },
      { path: '/account', element: <RequireAuth><SW><AccountPage /></SW></RequireAuth> },

      // Admin
      { path: '/admin/users', element: <RequireAuth adminOnly><SW><UsersPage /></SW></RequireAuth> },

      // 404
      { path: '*', element: <div className="flex flex-col items-center justify-center h-full gap-4"><h1 className="text-4xl font-bold text-text-primary">404</h1><p className="text-text-secondary">Page introuvable</p></div> },
    ],
  },
])
