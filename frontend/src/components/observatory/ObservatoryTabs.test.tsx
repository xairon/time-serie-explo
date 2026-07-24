import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import i18n from '@/i18n/config'
import { ObservatoryTabs } from './ObservatoryTabs'

function renderAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <ObservatoryTabs />
    </MemoryRouter>,
  )
}

describe('ObservatoryTabs — sous-onglets Observatoire (2026-07-24)', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('rend les deux sous-onglets : Nappes & rivières et Climat', () => {
    renderAt('/')
    expect(screen.getByRole('link', { name: 'Nappes & rivières' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Climat' })).toBeInTheDocument()
  })

  it('marque « Nappes & rivières » actif sur la route racine', () => {
    renderAt('/')
    expect(screen.getByRole('link', { name: 'Nappes & rivières' })).toHaveAttribute('aria-current', 'page')
    expect(screen.getByRole('link', { name: 'Climat' })).not.toHaveAttribute('aria-current')
  })

  it('marque « Climat » actif sur /climat', () => {
    renderAt('/climat')
    expect(screen.getByRole('link', { name: 'Climat' })).toHaveAttribute('aria-current', 'page')
    expect(screen.getByRole('link', { name: 'Nappes & rivières' })).not.toHaveAttribute('aria-current')
  })

  it('porte l’ancre de visite guidée nav-climat sur le sous-onglet Climat', () => {
    renderAt('/')
    expect(screen.getByRole('link', { name: 'Climat' })).toHaveAttribute('data-tour', 'nav-climat')
  })
})
