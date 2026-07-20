import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import i18n from '@/i18n/config'
import { SituationBanner } from './SituationBanner'
import type { ClimatSituationSummary } from '@/lib/observatory-types'

const summary: ClimatSituationSummary = {
  month: '2026-06', window: 3, n_cells: 100,
  classes_pct: { EXTREMEMENT_BAS: 10, TRES_BAS: 15, BAS: 20, NORMAL: 40, HAUT: 10, TRES_HAUT: 5, EXTREMEMENT_HAUT: 0 },
  pct_secheresse: 45, median_spi: -0.7, driest_since_year: 2011, is_driest_on_record: false,
  top5_cellules_seches: [], available: true,
}

describe('SituationBanner', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('affiche la barre de distribution mais plus de phrase de synthèse', () => {
    render(<SituationBanner summary={summary} isLoading={false} />)
    // La barre 7 classes reste — c'est un graphique, pas du texte narré.
    expect(screen.getByRole('img', { name: /distribution/i })).toBeInTheDocument()
    // La phrase auto-générée est retirée (« faisait trop LLM ») : ni le %,
    // ni le « mois le plus sec depuis AAAA ».
    expect(screen.queryByText(/45\s*%/)).not.toBeInTheDocument()
    expect(screen.queryByText(/du territoire/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/depuis/i)).not.toBeInTheDocument()
  })

  it("n'affiche plus de coordonnées lat/lon brutes", () => {
    render(<SituationBanner summary={summary} isLoading={false} />)
    expect(screen.queryByText(/°N|°O|°E|°S/)).not.toBeInTheDocument()
  })
})
