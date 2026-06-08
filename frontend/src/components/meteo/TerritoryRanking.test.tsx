import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import i18n from '@/i18n/config'
import { TerritoryRanking, sortByCriticality } from './TerritoryRanking'
import type { TerritorySituation } from '@/lib/observatory-types'

const mk = (code: string, cls: any, n = 10, prov = 0): TerritorySituation => ({
  level: 'region', code, name: `R${code}`, type: 'piezo', situation_class: cls,
  trend: 'baisse', pct_below_normal: 10, n_eligible: n, n_provisoire: prov,
  distribution: {}, insufficient: cls === null, outlook: null,
})

beforeAll(() => { i18n.changeLanguage('fr') })

describe('TerritoryRanking', () => {
  it('sorts driest-first and pushes insufficient to the end', () => {
    const sorted = sortByCriticality([mk('a', 'NORMAL'), mk('b', 'EXTREMEMENT_BAS'), mk('c', null)])
    expect(sorted.map(t => t.code)).toEqual(['b', 'a', 'c'])
  })
  it('renders rows', () => {
    render(<I18nextProvider i18n={i18n}><TerritoryRanking territories={[mk('b', 'TRES_BAS')]} onSelect={vi.fn()} /></I18nextProvider>)
    expect(screen.getByText('Rb')).toBeInTheDocument()
  })
  it('surfaces the provisoire (unclassified) station count', () => {
    render(<I18nextProvider i18n={i18n}><TerritoryRanking territories={[mk('z', 'BAS', 10, 3)]} onSelect={vi.fn()} /></I18nextProvider>)
    expect(screen.getByText('+3')).toBeInTheDocument()
    expect(screen.getByTitle(/3 stations non classées/)).toBeInTheDocument()
  })
})
