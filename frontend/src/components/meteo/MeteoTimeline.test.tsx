// frontend/src/components/meteo/MeteoTimeline.test.tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MeteoTimeline } from './MeteoTimeline'
import { addMonths, comparePeriods } from '@/lib/meteo-timeline'

function monthRange(start: string, end: string): string[] {
  const out: string[] = []
  let p = start
  while (comparePeriods(p, end) <= 0) { out.push(p); p = addMonths(p, 1) }
  return out
}

const periods = monthRange('2020-01', '2026-06')

describe('MeteoTimeline', () => {
  it('renders 15 month cells (12 data + 3 future greyed)', () => {
    render(<MeteoTimeline periods={periods} selected="2026-06" onChange={() => {}} />)
    const buttons = screen.getAllByRole('button', { name: /^mois / })
    expect(buttons).toHaveLength(15)
    // Future months are disabled
    expect(screen.getByRole('button', { name: 'mois juillet 2026' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'mois août 2026' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'mois septembre 2026' })).toBeDisabled()
  })

  it('clicking an available month fires onChange', () => {
    const onChange = vi.fn()
    render(<MeteoTimeline periods={periods} selected="2026-06" onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'mois mai 2026' }))
    expect(onChange).toHaveBeenCalledWith('2026-05')
  })

  it('shows the selected period in the date chip', () => {
    render(<MeteoTimeline periods={periods} selected="2026-06" onChange={() => {}} />)
    expect(screen.getByText('juin 2026')).toBeInTheDocument()
  })

  it('date chip × resets to the latest period', () => {
    const onChange = vi.fn()
    render(<MeteoTimeline periods={periods} selected="2024-03" onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'Revenir au mois le plus récent' }))
    expect(onChange).toHaveBeenCalledWith('2026-06')
  })
})
