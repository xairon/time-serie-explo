import { describe, it, expect, vi, beforeAll, afterAll } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import i18n from '@/i18n/config'
import { MonthStepper } from './MonthStepper'

describe('MonthStepper', () => {
  afterAll(async () => { await i18n.changeLanguage('fr') })

  it('renders French labels and month by default', async () => {
    await i18n.changeLanguage('fr')
    render(<MonthStepper month="2026-06" onChange={() => {}} />)
    expect(screen.getByRole('button', { name: 'Mois précédent' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Mois suivant' })).toBeInTheDocument()
    expect(screen.getByText('juin 2026')).toBeInTheDocument()
  })

  describe('in English', () => {
    beforeAll(async () => { await i18n.changeLanguage('en') })

    it('renders English aria-labels and an Intl-formatted month', () => {
      render(<MonthStepper month="2026-06" onChange={() => {}} />)
      expect(screen.getByRole('button', { name: 'Previous month' })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Next month' })).toBeInTheDocument()
      expect(screen.getByText('June 2026')).toBeInTheDocument()
    })

    it('calls onChange with the adjacent month when stepping', () => {
      const onChange = vi.fn()
      render(<MonthStepper month="2026-06" onChange={onChange} />)
      fireEvent.click(screen.getByRole('button', { name: 'Next month' }))
      expect(onChange).toHaveBeenCalledWith('2026-07')
      fireEvent.click(screen.getByRole('button', { name: 'Previous month' }))
      expect(onChange).toHaveBeenCalledWith('2026-05')
    })

    it('disables the next button at maxMonth', () => {
      render(<MonthStepper month="2026-06" onChange={() => {}} maxMonth="2026-06" />)
      expect(screen.getByRole('button', { name: 'Next month' })).toBeDisabled()
      expect(screen.getByRole('button', { name: 'Previous month' })).not.toBeDisabled()
    })
  })
})
