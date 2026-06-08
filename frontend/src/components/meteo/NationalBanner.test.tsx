import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import i18n from '@/i18n/config'
import { NationalBanner } from './NationalBanner'

beforeAll(() => { i18n.changeLanguage('fr') })

const base = {
  type: 'piezo' as const, situation_class: 'BAS' as const, trend: 'baisse' as const,
  pct_below_normal: 42, n_eligible: 900, n_provisoire: 100, distribution: {}, insufficient: false, outlook: null,
}

const renderWith = (ui: React.ReactElement) =>
  render(<I18nextProvider i18n={i18n}>{ui}</I18nextProvider>)

describe('NationalBanner', () => {
  it('shows the verdict class label and headline number', () => {
    renderWith(<NationalBanner data={base} departmentsInAlert={7} />)
    expect(screen.getByText(/Bas/)).toBeInTheDocument()
    expect(screen.getByText(/42 % sous la normale/)).toBeInTheDocument()
  })
  it('renders an insufficient state without a class', () => {
    renderWith(<NationalBanner data={{ ...base, situation_class: null, insufficient: true }} departmentsInAlert={0} />)
    expect(screen.getByText(/Données insuffisantes/)).toBeInTheDocument()
  })
})
