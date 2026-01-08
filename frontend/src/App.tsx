/**
 * Main App component with routing between pages
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Layout } from './components/Layout'
import { useAppStore } from './store/appStore'
import { DatasetsPage } from './pages/DatasetsPage'
import { TrainingPage } from './pages/TrainingPage'
import { ForecastingPage } from './pages/ForecastingPage'
import { ModelsPage } from './pages/ModelsPage'
import './index.css'

// Create query client for react-query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
    },
  },
})

function AppContent() {
  const activePage = useAppStore((state) => state.activePage)

  // Render active page
  const renderPage = () => {
    switch (activePage) {
      case 'datasets':
        return <DatasetsPage />
      case 'training':
        return <TrainingPage />
      case 'forecasting':
        return <ForecastingPage />
      case 'models':
        return <ModelsPage />
      default:
        return <DatasetsPage />
    }
  }

  return (
    <Layout>
      {renderPage()}
    </Layout>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  )
}

export default App
