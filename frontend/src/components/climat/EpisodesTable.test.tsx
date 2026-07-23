import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen, fireEvent, within } from '@testing-library/react'
import i18n from '@/i18n/config'
import { EpisodesTable } from './EpisodesTable'
import type { ClimatDroughtEpisode } from '@/lib/observatory-types'

const EPISODES: ClimatDroughtEpisode[] = [
  { debut: '1976-04-01', fin: '1976-08-01', duree_mois: 5, index_min: -2.1, deficit_cumule_mm: -180.4 },
  { debut: '2003-06-01', fin: '2003-09-01', duree_mois: 4, index_min: -1.9, deficit_cumule_mm: -120.0 },
]

function rows() {
  return screen.getAllByRole('row').slice(1) // drop header row
}

describe('EpisodesTable', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('shows a skeleton while loading', () => {
    const { container } = render(<EpisodesTable episodes={[]} isLoading />)
    expect(container.querySelectorAll('.animate-pulse').length).toBeGreaterThan(0)
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('shows the empty state when there are no episodes', () => {
    render(<EpisodesTable episodes={[]} isLoading={false} />)
    expect(screen.getByText('Aucun épisode de sécheresse recensé sur cette période.')).toBeInTheDocument()
  })

  it('renders one row per episode, sorted by duration desc by default', () => {
    render(<EpisodesTable episodes={EPISODES} isLoading={false} />)
    const rendered = rows()
    expect(rendered).toHaveLength(2)
    // Longest episode (5 months, 1976) first.
    expect(within(rendered[0]).getByText('5')).toBeInTheDocument()
    expect(within(rendered[1]).getByText('4')).toBeInTheDocument()
  })

  it('re-sorts when a sortable header is clicked, and reverses on a second click', () => {
    render(<EpisodesTable episodes={EPISODES} isLoading={false} />)
    // First click on a newly-active column sorts descending: 2003 (later start) first.
    fireEvent.click(screen.getByRole('button', { name: /Début/ }))
    let rendered = rows()
    expect(within(rendered[0]).getByText('4')).toBeInTheDocument()
    // Second click reverses to ascending: 1976 first.
    fireEvent.click(screen.getByRole('button', { name: /Début/ }))
    rendered = rows()
    expect(within(rendered[0]).getByText('5')).toBeInTheDocument()
  })

  it('highlights the current episode with an "En cours" badge', () => {
    render(<EpisodesTable episodes={EPISODES} isLoading={false} currentEpisode={EPISODES[1]} />)
    expect(screen.getByText('En cours')).toBeInTheDocument()
    const rendered = rows()
    const currentRow = rendered.find((r) => within(r).queryByText('En cours'))
    expect(currentRow?.className).toContain('bg-amber-500/10')
  })

  it('does not highlight anything when there is no current episode', () => {
    render(<EpisodesTable episodes={EPISODES} isLoading={false} />)
    expect(screen.queryByText('En cours')).not.toBeInTheDocument()
  })

  it('formats the SPI min and deficit values', () => {
    render(<EpisodesTable episodes={[EPISODES[0]]} isLoading={false} />)
    expect(screen.getByText('-2.10')).toBeInTheDocument()
    expect(screen.getByText('-180 mm')).toBeInTheDocument()
  })

  it('shows the SPI min header by default', () => {
    render(<EpisodesTable episodes={EPISODES} isLoading={false} />)
    expect(screen.getByText('SPI min')).toBeInTheDocument()
    expect(screen.queryByText('Min SPEI')).not.toBeInTheDocument()
  })

  it('shows the SPEI min header when index="spei"', () => {
    render(<EpisodesTable episodes={EPISODES} isLoading={false} index="spei" />)
    expect(screen.getByText('Min SPEI')).toBeInTheDocument()
    expect(screen.queryByText('SPI min')).not.toBeInTheDocument()
  })
})
