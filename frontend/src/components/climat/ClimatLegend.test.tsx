import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import i18n from '@/i18n/config'
import { ClimatLegend } from './ClimatLegend'

describe('ClimatLegend — pluie journalière', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('affiche les bornes en mm, sans noms de classes inventés', () => {
    render(<ClimatLegend variable="precip_daily" window={1} month="2025-06-15" />)
    // La valeur EST le sens : on montre les bornes, pas un « faible/fort » éditorialisé.
    // getAllByText (pas getByText) : la borne 0.1 apparaît à la fois dans la classe
    // sèche (« < 0,1 ») et dans la borne de la classe suivante (« 0.1 – 1 ») — les deux
    // sont un rendu correct, pas un doublon fautif.
    expect(screen.getAllByText(/0,1|0\.1/).length).toBeGreaterThan(0)
    expect(screen.getByText(/≥\s*50/)).toBeInTheDocument()
    expect(screen.queryByText(/faible|modéré|fort/i)).not.toBeInTheDocument()
  })
})
