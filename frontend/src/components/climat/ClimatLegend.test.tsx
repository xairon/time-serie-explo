import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import i18n from '@/i18n/config'
import { ClimatLegend } from './ClimatLegend'

describe('ClimatLegend — pluie journalière', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('affiche les bornes en mm, sans noms de classes inventés', () => {
    render(<ClimatLegend variable="precip_daily" window={1} month="2025-06-15" />)
    // La valeur EST le sens : on montre les bornes, pas un « faible/fort » éditorialisé.
    // Classe sèche : assertion exacte (climat.legend.precipDryClass = « < 0,1 »), pas
    // une regex large — sinon le test passerait encore si cette classe disparaissait,
    // la bande intérieure « 0.1 – 1 » suffisant seule à la satisfaire.
    expect(screen.getByText('< 0,1 mm')).toBeInTheDocument()
    // Classe la plus haute, distincte de la classe sèche.
    expect(screen.getByText('≥ 50 mm')).toBeInTheDocument()
    expect(screen.queryByText(/faible|modéré|fort/i)).not.toBeInTheDocument()
  })
})
