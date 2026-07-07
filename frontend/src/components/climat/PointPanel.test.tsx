import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import i18n from '@/i18n/config'
import { PointPanel } from './PointPanel'
import type { ClimatPointSeriesEntry, ClimatDroughtEpisode } from '@/lib/observatory-types'

vi.mock('./PrecipNormalChart', () => ({ PrecipNormalChart: () => <div data-testid="precip-chart" /> }))
vi.mock('./ClimatIndexChart', () => ({ ClimatIndexChart: () => <div data-testid="index-chart" /> }))
vi.mock('./CompareYearsSection', () => ({ CompareYearsSection: () => <div data-testid="compare-section" /> }))
vi.mock('@/hooks/useClimat', () => ({
  useClimatPointSeries: vi.fn(),
  useClimatPointEpisodes: vi.fn(),
  EPISODES_WINDOW: 3,
}))

import { useClimatPointSeries, useClimatPointEpisodes } from '@/hooks/useClimat'

const mockSeriesHook = useClimatPointSeries as unknown as ReturnType<typeof vi.fn>
const mockEpisodesHook = useClimatPointEpisodes as unknown as ReturnType<typeof vi.fn>

const SERIES: ClimatPointSeriesEntry[] = [
  { month: '2026-04-01', temperature_moyenne: 12, temperature_min: 8, temperature_max: 16, precipitation_totale: 40, etp_totale: 50, bilan_hydrique: -10, nb_jours: 30, mois_complet: true, precipitation_normale: 60, temperature_normale: 11, spi_1: -0.2, sti_1: 0.1, spi_3: -1.4, sti_3: 0.3, spi_6: -0.9, sti_6: 0.2, spi_12: -0.5, sti_12: 0.1 },
  { month: '2026-05-01', temperature_moyenne: 14, temperature_min: 9, temperature_max: 18, precipitation_totale: 20, etp_totale: 60, bilan_hydrique: -40, nb_jours: 31, mois_complet: true, precipitation_normale: 55, temperature_normale: 13, spi_1: -1.6, sti_1: 0.4, spi_3: -1.8, sti_3: 0.5, spi_6: -1.1, sti_6: 0.3, spi_12: -0.6, sti_12: 0.2 },
]

const EPISODES: ClimatDroughtEpisode[] = [
  { debut: '2026-03-01', fin: '2026-05-01', duree_mois: 3, spi_min: -1.8, deficit_cumule_mm: -90.5 },
]

function mockSuccess(series = SERIES, episodes = EPISODES) {
  mockSeriesHook.mockReturnValue({ data: { cell: { latitude: 47.4, longitude: 0.7 }, series }, isLoading: false, isError: false })
  mockEpisodesHook.mockReturnValue({ data: episodes, isLoading: false, isError: false })
}

describe('PointPanel', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('shows the title and formatted coordinates', () => {
    mockSuccess()
    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)
    expect(screen.getByText('Analyse du point')).toBeInTheDocument()
    expect(screen.getByText('47.40° N, 0.70° E')).toBeInTheDocument()
  })

  it('calls onClose when the close button is clicked', () => {
    mockSuccess()
    const onClose = vi.fn()
    render(<PointPanel lat={47.4} lon={0.7} onClose={onClose} />)
    fireEvent.click(screen.getByRole('button', { name: 'Fermer' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('calls onClose on Escape', () => {
    mockSuccess()
    const onClose = vi.fn()
    render(<PointPanel lat={47.4} lon={0.7} onClose={onClose} />)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('renders a direct download link to the CSV export', () => {
    mockSuccess()
    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)
    const link = screen.getByRole('link', { name: /Exporter en CSV/ })
    expect(link).toHaveAttribute('href', expect.stringContaining('/observatory/climat/export-point.csv?lat=47.4&lon=0.7'))
    expect(link).toHaveAttribute('download')
  })

  it('renders the charts and episodes table once data is loaded', () => {
    mockSuccess()
    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)
    expect(screen.getByTestId('precip-chart')).toBeInTheDocument()
    expect(screen.getByTestId('index-chart')).toBeInTheDocument()
    expect(screen.getByText('Épisodes de sécheresse')).toBeInTheDocument()
  })

  it('renders the Comparaison section regardless of the point-series load state', () => {
    mockSuccess()
    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)
    expect(screen.getByTestId('compare-section')).toBeInTheDocument()
  })

  it('shows a loading state while the series is loading', () => {
    mockSeriesHook.mockReturnValue({ data: undefined, isLoading: true, isError: false })
    mockEpisodesHook.mockReturnValue({ data: undefined, isLoading: true, isError: false })
    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)
    expect(screen.queryByTestId('precip-chart')).not.toBeInTheDocument()
  })

  it('shows an error message when the point series fails to load', () => {
    mockSeriesHook.mockReturnValue({ data: undefined, isLoading: false, isError: true })
    mockEpisodesHook.mockReturnValue({ data: undefined, isLoading: false, isError: false })
    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)
    expect(screen.getByText('Impossible de charger les données de ce point.')).toBeInTheDocument()
    expect(screen.queryByTestId('precip-chart')).not.toBeInTheDocument()
  })

  it('highlights the ongoing episode when the last month is in drought (spi_3 < -1)', () => {
    mockSuccess()
    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)
    // SERIES' last entry (2026-05) has spi_3 = -1.8 and EPISODES' last episode ends 2026-05-01.
    expect(screen.getByText('En cours')).toBeInTheDocument()
  })

  it('does not highlight any episode when the last month is not in drought', () => {
    const calmSeries = SERIES.map((s, i) => (i === SERIES.length - 1 ? { ...s, spi_3: 0.2 } : s))
    mockSuccess(calmSeries)
    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)
    expect(screen.queryByText('En cours')).not.toBeInTheDocument()
  })
})
