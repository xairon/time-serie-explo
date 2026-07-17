import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import i18n from '@/i18n/config'
import ClimatPage from './ClimatPage'

// Deux couches journalières, deux plages de couverture DIFFÉRENTES (comme en
// production : température -> 2026-07-10, pluie -> 2026-07-12). Le bug régressif
// (Finding 1) : passer de la pluie (jour le plus récent, 07-12) à Tx laissait
// `s.day` bloqué à 07-12, hors bornes de la plage température.
// vi.mock factories are hoisted above imports/consts — vi.hoisted keeps these
// two literals reachable from inside the factory below.
const { TEMP_RANGE, PRECIP_RANGE } = vi.hoisted(() => ({
  TEMP_RANGE: { min_date: '1950-01-01', max_date: '2026-07-10' },
  PRECIP_RANGE: { min_date: '1950-01-01', max_date: '2026-07-12' },
}))

vi.mock('@/components/climat/ClimatMap', () => ({ ClimatMap: () => <div data-testid="climat-map" /> }))
vi.mock('@/components/climat/SituationBanner', () => ({ SituationBanner: () => <div data-testid="situation-banner" /> }))
vi.mock('@/components/climat/DailyTempBanner', () => ({ DailyTempBanner: () => <div data-testid="daily-temp-banner" /> }))
vi.mock('@/components/climat/PointPanel', () => ({ PointPanel: () => <div data-testid="point-panel" /> }))
vi.mock('@/components/climat/ClimatLegend', () => ({ ClimatLegend: () => <div data-testid="climat-legend" /> }))

vi.mock('@/hooks/useClimat', () => ({
  useClimatGridMonthly: vi.fn().mockReturnValue({ data: [], isLoading: false }),
  useClimatGridIndices: vi.fn().mockReturnValue({ data: [], isLoading: false }),
  useClimatSituationSummary: vi.fn().mockReturnValue({ data: undefined, isLoading: false }),
  useClimatRange: vi.fn().mockReturnValue({
    data: { max_indices_month: '2026-06', max_monthly_month: '2026-07', min_month: '1950-01' },
  }),
  useSelectedCellParam: vi.fn().mockReturnValue({ selectedCell: null, selectCell: vi.fn(), clearSelectedCell: vi.fn() }),
  useClimatDailyTempRange: vi.fn().mockReturnValue({ data: TEMP_RANGE }),
  useClimatDailyTemp: vi.fn().mockReturnValue({ data: [], isLoading: false }),
  useClimatDailyPrecip: vi.fn().mockReturnValue({ data: [], isLoading: false }),
  useClimatDailyPrecipRange: vi.fn().mockReturnValue({ data: PRECIP_RANGE }),
}))

function renderPage() {
  render(<MemoryRouter><ClimatPage /></MemoryRouter>)
}

describe('ClimatPage — jour hors bornes au changement de variable journalière (Finding 1)', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('ramène le jour dans les bornes de la nouvelle plage au lieu de le laisser hors bornes', () => {
    renderPage()

    // `s.day` se pose sur 2026-07-10 dès le montage (la plage journalière active
    // au premier rendu est celle de la température, tant qu'aucune variable
    // journalière n'a été choisie) — comportement de défaut inchangé, cf.
    // resolveDefaultDay. Bascule sur « Pluie (jour) » : 07-10 reste dans les
    // bornes pluie (max 07-12), donc pas de reset — comportement voulu : on ne
    // saute pas vers le max juste parce que la famille change.
    fireEvent.click(screen.getByRole('radio', { name: 'Pluie (jour)' }))
    expect(screen.getByText('10 juillet 2026')).toBeInTheDocument()

    // L'utilisateur avance de deux jours, jusqu'au 07-12 — valide pour la pluie,
    // hors de portée pour la température.
    const nextDay = screen.getByRole('button', { name: 'Jour suivant' })
    fireEvent.click(nextDay)
    fireEvent.click(nextDay)
    expect(screen.getByText('12 juillet 2026')).toBeInTheDocument()

    // Bascule sur Tx : la plage active devient TEMP_RANGE (max 07-10). Avant le
    // correctif, `s.day` restait à 07-12 (hors bornes) — carte vide, "Suivant" grisé.
    fireEvent.click(screen.getByRole('radio', { name: 'Tx (max)' }))
    expect(screen.getByText('10 juillet 2026')).toBeInTheDocument()
    expect(screen.queryByText('12 juillet 2026')).not.toBeInTheDocument()
  })
})
