import { describe, it, expect, beforeAll, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import i18n from '@/i18n/config'
import { CompareSelectionProvider } from '@/contexts/CompareSelection'
import { TopNav } from './TopNav'

// TopNav pulls in live health/auth hooks that hit the network via
// fetch — irrelevant to what this test checks (the nav items themselves) and
// slow/flaky under jsdom, so both are stubbed out.
vi.mock('@/hooks/useHealth', () => ({ useHealth: () => ({ data: { status: 'ok' } }) }))
vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => ({ user: null, logout: vi.fn() }),
}))

function renderNav() {
  render(
    <MemoryRouter>
      <CompareSelectionProvider>
        <TopNav />
      </CompareSelectionProvider>
    </MemoryRouter>,
  )
}

describe('TopNav — Climat promu en sous-onglet Observatoire (2026-07-24)', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('ne contient plus Climat dans la barre principale', () => {
    renderNav()
    expect(screen.queryByRole('link', { name: /climat/i })).not.toBeInTheDocument()
  })

  it('rend les 4 entrées restantes : Observatoire, Comparer, Pastas Lab, AI Lab', () => {
    renderNav()
    expect(screen.getByRole('link', { name: /observatoire/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /comparer/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /pastas lab/i })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /lab ia/i })).toBeInTheDocument()

    // 4 entrées dans la barre desktop (le menu mobile, masqué, duplique les
    // mêmes labels — on ne compte donc que les liens visibles au clavier/lecteur
    // d'écran par rôle, ce qui inclut les deux listes ; on vérifie donc le total
    // exact pour être sûr qu'aucune 5e entrée (Climat) n'a survécu ailleurs).
    const allLinks = screen.getAllByRole('link')
    const navLabels = allLinks.map((l) => l.textContent).filter((t): t is string => !!t)
    expect(navLabels.some((t) => /climat/i.test(t))).toBe(false)
  })
})
