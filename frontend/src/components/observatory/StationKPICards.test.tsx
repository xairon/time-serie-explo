import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import i18n from '@/i18n/config'
import { StationKPICards } from './StationKPICards'
import { SPI_CLASS_COLORS } from '@/lib/era5-colors'
import type { SPIDataPoint } from '@/lib/observatory-types'

const PIEZO_STATION = {
  code_bss: 'BSS000TEST',
  niveau_derniere_annee: 102.4,
  niveau_moyen_global: 101.9,
  precipitation_moyenne_mensuelle: 63.2,
  temperature_moyenne_globale: 11.8,
}

const SPI3: SPIDataPoint = { mois: '2026-05-01', value: 12.3, spi: -1.35, classification: 'BAS' }

describe('StationKPICards — SPI (3 mois) tile', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('renders the SPI tile with the value and the class badge', () => {
    render(<StationKPICards station={PIEZO_STATION} type="piezo" spi3={SPI3} />)
    expect(screen.getByText('SPI (3 mois)')).toBeInTheDocument()
    expect(screen.getByText('-1.35')).toBeInTheDocument()
    const badge = screen.getByText('Modérément sec') // observatory.spi.BAS
    expect(badge).toBeInTheDocument()
    expect(badge).toHaveStyle({ backgroundColor: SPI_CLASS_COLORS['BAS'] })
  })

  it('does not render the tile when no SPI point is available', () => {
    render(<StationKPICards station={PIEZO_STATION} type="piezo" spi3={null} />)
    expect(screen.queryByText('SPI (3 mois)')).not.toBeInTheDocument()
  })

  it('does not render the tile when the prop is omitted (backward compatible)', () => {
    render(<StationKPICards station={PIEZO_STATION} type="piezo" />)
    expect(screen.queryByText('SPI (3 mois)')).not.toBeInTheDocument()
  })

  it('shows a placeholder value but still the badge when spi is null on the latest point', () => {
    render(<StationKPICards station={PIEZO_STATION} type="piezo" spi3={{ ...SPI3, spi: null, classification: 'UNKNOWN' }} />)
    expect(screen.getByText('SPI (3 mois)')).toBeInTheDocument()
    expect(screen.getByText('--')).toBeInTheDocument()
  })

  it('keeps the existing piezo tiles', () => {
    render(<StationKPICards station={PIEZO_STATION} type="piezo" spi3={SPI3} />)
    expect(screen.getByText('État actuel')).toBeInTheDocument()
    expect(screen.getByText('Précipitation moy.')).toBeInTheDocument()
    expect(screen.getByText('Température moy.')).toBeInTheDocument()
  })
})
