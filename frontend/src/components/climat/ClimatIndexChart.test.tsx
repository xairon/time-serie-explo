import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import i18n from '@/i18n/config'
import { ClimatIndexChart } from './ClimatIndexChart'
import type { ClimatPointSeriesEntry } from '@/lib/observatory-types'

const NO_DATA_TEXT = "Aucune donnée d'indice pour cette période."

/** Minimal point-series entry — only the fields the chart reads. */
function entry(month: string, overrides: Partial<ClimatPointSeriesEntry> = {}): ClimatPointSeriesEntry {
  return {
    month,
    temperature_moyenne: null, temperature_min: null, temperature_max: null,
    precipitation_totale: null, etp_totale: null, bilan_hydrique: null, nb_jours: null,
    mois_complet: true, precipitation_normale: null, temperature_normale: null,
    spi_1: null, sti_1: null, spei_1: null, spi_3: null, sti_3: null, spei_3: null,
    spi_6: null, sti_6: null, spei_6: null, spi_12: null, sti_12: null, spei_12: null,
    ...overrides,
  }
}

describe('ClimatIndexChart — SPEI toggle', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('offers a SPEI toggle alongside SPI and STI', () => {
    const series = [entry('2026-05-01', { spi_3: -1.2 })]
    render(<ClimatIndexChart series={series} window={3} onWindowChange={() => {}} />)
    expect(screen.getByRole('radio', { name: 'SPI (précipitations)' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'STI (température)' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'SPEI (précip. − ETP)' })).toBeInTheDocument()
  })

  it('reads spei_{window} once SPEI is selected — not spi_{window} or sti_{window}', () => {
    // spi_3/sti_3 carry data but spei_3 is null: if the chart fell back to
    // another field it would render data; reading spei_3 correctly shows the
    // "no data" placeholder instead.
    const series = [entry('2026-05-01', { spi_3: -1.2, sti_3: 0.4, spei_3: null })]
    render(<ClimatIndexChart series={series} window={3} onWindowChange={() => {}} />)

    // SPI (default) has data — no placeholder.
    expect(screen.queryByText(NO_DATA_TEXT)).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('radio', { name: 'SPEI (précip. − ETP)' }))
    expect(screen.getByText(NO_DATA_TEXT)).toBeInTheDocument()
  })

  it('renders SPEI data (drops the placeholder) once spei_{window} is populated', () => {
    // spi_3 must also be populated so the toggle itself renders on the default
    // ('spi') selection — the chart replaces the whole toggle row with the
    // placeholder when the ACTIVE index has no data for the period.
    const series = [entry('2026-05-01', { spi_3: -1.0, spei_3: -1.8 })]
    render(<ClimatIndexChart series={series} window={3} onWindowChange={() => {}} />)

    fireEvent.click(screen.getByRole('radio', { name: 'SPEI (précip. − ETP)' }))
    expect(screen.queryByText(NO_DATA_TEXT)).not.toBeInTheDocument()
  })

  it('switches to spei_{window} for the currently selected window (not a hardcoded one)', () => {
    const series = [entry('2026-05-01', { spi_6: -1.0, spei_3: null, spei_6: -1.5 })]
    render(<ClimatIndexChart series={series} window={6} onWindowChange={() => {}} />)

    fireEvent.click(screen.getByRole('radio', { name: 'SPEI (précip. − ETP)' }))
    // window prop is 6 → fieldKey must be spei_6 (populated), not spei_3 (null).
    expect(screen.queryByText(NO_DATA_TEXT)).not.toBeInTheDocument()
  })
})
