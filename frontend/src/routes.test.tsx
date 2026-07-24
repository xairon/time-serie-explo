import { describe, it, expect, beforeAll, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { RouterProvider } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import i18n from '@/i18n/config'
import { router } from './routes'

// ObservatoryPage/ClimatPage drag in maplibre-gl and a full data-fetching
// stack — irrelevant to a routing test, so both are stubbed to a tagged div.
// The real router config from routes.tsx is exercised as-is (not a hand-rolled
// copy), so this catches a regression in the actual nesting.
vi.mock('@/pages/ObservatoryPage', () => ({ default: () => <div data-testid="observatory-page" /> }))
vi.mock('@/pages/ClimatPage', () => ({ default: () => <div data-testid="climat-page" /> }))
vi.mock('@/hooks/useHealth', () => ({ useHealth: () => ({ data: { status: 'ok' } }) }))
vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => ({ user: null, logout: vi.fn() }),
}))

function renderRouter() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  )
}

describe('routes — Climat en sous-onglet Observatoire, /climat reste un deep-link valide (2026-07-24)', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('rend ObservatoryPage à la racine, sous la coquille Observatoire (sous-onglets visibles)', async () => {
    await router.navigate('/')
    renderRouter()
    await waitFor(() => expect(screen.getByTestId('observatory-page')).toBeInTheDocument())
    // ObservatoryShell renders ObservatoryTabs above the routed page.
    expect(screen.getByRole('link', { name: 'Nappes & rivières' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Climat' })).toBeInTheDocument()
    expect(screen.queryByTestId('climat-page')).not.toBeInTheDocument()
  })

  it('non-régression deep-link : /climat rend toujours ClimatPage, à l’intérieur de la coquille', async () => {
    await router.navigate('/climat')
    renderRouter()
    await waitFor(() => expect(screen.getByTestId('climat-page')).toBeInTheDocument())
    expect(screen.queryByTestId('observatory-page')).not.toBeInTheDocument()
    // The sub-tab bar is still there and marks Climat active.
    expect(screen.getByRole('link', { name: 'Climat' })).toHaveAttribute('aria-current', 'page')
  })
})
