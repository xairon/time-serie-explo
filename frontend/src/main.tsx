import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './i18n/config'
import App from './App'

const rootElement = document.getElementById('root')
if (!rootElement) {
  throw new Error('Root element not found. Ensure there is a <div id="root"> in index.html.')
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Warm the heavy map chunk (vendor-maplibre) during idle time so the observatory
// map renders without waiting on a separate download after navigation.
const warmMap = () => { import('maplibre-gl') }
if ('requestIdleCallback' in window) {
  ;(window as Window & typeof globalThis).requestIdleCallback(warmMap)
} else {
  setTimeout(warmMap, 2000)
}
