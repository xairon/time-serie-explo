import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import i18n from '@/i18n/config'
import { CompareSpiMiniMaps } from './CompareSpiMiniMaps'

vi.mock('@/hooks/useClimat', () => ({
  useClimatCompareGridIndices: vi.fn(),
}))

import { useClimatCompareGridIndices } from '@/hooks/useClimat'

const mockHook = useClimatCompareGridIndices as unknown as ReturnType<typeof vi.fn>

describe('CompareSpiMiniMaps', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('renders a mini-map per year once loaded', () => {
    mockHook.mockReturnValue([
      { data: [{ latitude: 47, longitude: 0.5, spi: -0.2, index_class: 'NORMAL' }], isLoading: false, isError: false },
      { data: [{ latitude: 47, longitude: 0.5, spi: -1.8, index_class: 'SEVERE_DROUGHT' }], isLoading: false, isError: false },
    ])
    render(<CompareSpiMiniMaps years={[2003, 2022]} month={6} onMonthChange={() => {}} />)
    expect(screen.getAllByTestId('mini-spi-map')).toHaveLength(2)
    expect(screen.queryByText('Données indisponibles')).not.toBeInTheDocument()
  })

  it('renders an error tile for a year whose query failed', () => {
    mockHook.mockReturnValue([
      { data: [{ latitude: 47, longitude: 0.5, spi: -0.2, index_class: 'NORMAL' }], isLoading: false, isError: false },
      { data: undefined, isLoading: false, isError: true },
    ])
    render(<CompareSpiMiniMaps years={[2003, 2022]} month={6} onMonthChange={() => {}} />)
    expect(screen.getAllByTestId('mini-spi-map')).toHaveLength(1)
    expect(screen.getByText('Données indisponibles')).toBeInTheDocument()
    expect(screen.getByText('2022')).toBeInTheDocument()
  })

  it('renders an error tile when a query succeeds without data', () => {
    mockHook.mockReturnValue([
      { data: undefined, isLoading: false, isError: false },
    ])
    render(<CompareSpiMiniMaps years={[2022]} month={6} onMonthChange={() => {}} />)
    expect(screen.queryByTestId('mini-spi-map')).not.toBeInTheDocument()
    expect(screen.getByText('Données indisponibles')).toBeInTheDocument()
  })
})
