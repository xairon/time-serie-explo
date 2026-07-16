import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import i18n from '@/i18n/config'
import { PointPanel } from './PointPanel'
import type { ClimatPointSeriesEntry, ClimatDroughtEpisode } from '@/lib/observatory-types'

vi.mock('./PrecipNormalChart', () => ({ PrecipNormalChart: () => <div data-testid="precip-chart" /> }))
vi.mock('./ClimatIndexChart', () => ({
  // A minimal stand-in that still exposes the real onWindowChange wiring, so
  // tests can prove the "en cours" highlight actually follows the selected
  // window (not just the hardcoded EPISODES_WINDOW default).
  ClimatIndexChart: ({ window, onWindowChange }: { window: number; onWindowChange: (w: number) => void }) => (
    <div data-testid="index-chart">
      <span data-testid="index-chart-window">{window}</span>
      <button onClick={() => onWindowChange(6)}>window-6</button>
    </div>
  ),
}))
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

  it('highlights the ongoing episode past a trailing null-spi month (partial current month)', () => {
    // 2026-06 stands in for the partial current month: no SPI computed yet
    // (null), same as production right after the daily grid rolls into a new
    // month. The real episode ends 2026-05, the last month WITH an spi_3.
    const seriesWithPartialMonth: ClimatPointSeriesEntry[] = [
      ...SERIES,
      { ...SERIES[SERIES.length - 1], month: '2026-06-01', spi_1: null, spi_3: null, spi_6: null, spi_12: null },
    ]
    const episodesEndingOnLastRealMonth: ClimatDroughtEpisode[] = [
      { debut: '2026-04-01', fin: '2026-05-01', duree_mois: 2, spi_min: -1.8, deficit_cumule_mm: -90.5 },
    ]
    mockSuccess(seriesWithPartialMonth, episodesEndingOnLastRealMonth)
    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)
    // A naive `series[series.length - 1]` read would see 2026-06's null spi_3
    // and never highlight anything — this must still fire.
    expect(screen.getByText('En cours')).toBeInTheDocument()
  })

  it('judges the ongoing highlight on spi_6 (not spi_3) once window 6 is selected', () => {
    // At the default window (3), the last month is calm (spi_3 = 0.3) — no
    // highlight. At window 6, the same last month is in a live drought
    // (spi_6 = -2.1) — proves the highlight reads the field for the SELECTED
    // window, not a hardcoded one.
    const series: ClimatPointSeriesEntry[] = [
      { ...SERIES[0], spi_3: -0.2, spi_6: -1.9 },
      { ...SERIES[1], spi_3: 0.3, spi_6: -2.1 },
    ]
    const episodes: ClimatDroughtEpisode[] = [
      { debut: '2026-04-01', fin: '2026-05-01', duree_mois: 2, spi_min: -2.1, deficit_cumule_mm: -80 },
    ]
    mockSuccess(series, episodes)
    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)
    expect(screen.queryByText('En cours')).not.toBeInTheDocument()

    fireEvent.click(screen.getByText('window-6'))
    expect(screen.getByText('En cours')).toBeInTheDocument()
  })

  it('refetches episodes at the new window and threads it through the table and the ongoing highlight', () => {
    // Distinct episode lists per window prove the table reflects a genuine
    // refetch (not just the highlight re-reading a static payload): window 3
    // has an old, closed episode; window 6 has a different, ongoing one.
    const episodesByWindow: Record<number, ClimatDroughtEpisode[]> = {
      3: [{ debut: '2026-02-01', fin: '2026-03-01', duree_mois: 1, spi_min: -1.5, deficit_cumule_mm: -50 }],
      6: [{ debut: '2026-04-01', fin: '2026-05-01', duree_mois: 2, spi_min: -2.1, deficit_cumule_mm: -80 }],
    }
    const series: ClimatPointSeriesEntry[] = [
      { ...SERIES[0], spi_3: -0.2, spi_6: -1.9 },
      { ...SERIES[1], spi_3: 0.3, spi_6: -2.1 }, // last month: calm at window 3, in drought at window 6
    ]
    mockSeriesHook.mockReturnValue({ data: { cell: { latitude: 47.4, longitude: 0.7 }, series }, isLoading: false, isError: false })
    mockEpisodesHook.mockImplementation((_lat: number, _lon: number, window: number) => ({
      data: episodesByWindow[window] ?? [],
      isLoading: false,
      isError: false,
    }))

    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)

    // Initial render: default window (3) — hook called with window 3, table shows the window-3 episode only.
    expect(mockEpisodesHook).toHaveBeenLastCalledWith(47.4, 0.7, 3)
    expect(screen.getByText('-50 mm')).toBeInTheDocument()
    expect(screen.queryByText('-80 mm')).not.toBeInTheDocument()
    expect(screen.queryByText('En cours')).not.toBeInTheDocument()

    fireEvent.click(screen.getByText('window-6'))

    // After the window change: hook re-invoked with the NEW window (refetch), the
    // table now shows the window-6 episode instead, and the highlight follows spi_6.
    expect(mockEpisodesHook).toHaveBeenLastCalledWith(47.4, 0.7, 6)
    expect(screen.getByText('-80 mm')).toBeInTheDocument()
    expect(screen.queryByText('-50 mm')).not.toBeInTheDocument()
    expect(screen.getByText('En cours')).toBeInTheDocument()
  })

  it('affiche le bilan du mois avec les vraies valeurs du dernier mois', () => {
    vi.mocked(useClimatPointSeries).mockReturnValue({
      data: { series: [
        { month: '2026-05', temperature_moyenne: 15.1, precipitation_totale: 70,
          etp_totale: 90, bilan_hydrique: -20 },
        { month: '2026-06', temperature_moyenne: 18.3, precipitation_totale: 40,
          etp_totale: 120, bilan_hydrique: -80 },
      ] },
      isLoading: false, isError: false,
    } as any)

    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)

    expect(screen.getByText('18.3 °C')).toBeInTheDocument()
    expect(screen.getByText('40 mm')).toBeInTheDocument()
    expect(screen.getByText('120 mm')).toBeInTheDocument()
    expect(screen.getByText('−80 mm')).toBeInTheDocument()   // U+2212, pas un tiret ASCII
    expect(screen.getByText('Déficit')).toBeInTheDocument()   // classifyBilan(-80) -> TRES_BAS
  })

  it('rend « — » sur les champs nuls d’un mois partiel sans masquer le bloc', () => {
    vi.mocked(useClimatPointSeries).mockReturnValue({
      data: { series: [
        { month: '2026-07', temperature_moyenne: null, precipitation_totale: null,
          etp_totale: null, bilan_hydrique: null },
      ] },
      isLoading: false, isError: false,
    } as any)

    render(<PointPanel lat={47.4} lon={0.7} onClose={() => {}} />)

    expect(screen.getByText('Bilan du mois')).toBeInTheDocument()
    expect(screen.getAllByText('—')).toHaveLength(4)
  })
})
