import { describe, it, expect, vi, beforeAll, afterAll } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import i18n from '@/i18n/config'
import { DayStepper } from './DayStepper'

describe('DayStepper', () => {
  afterAll(async () => { await i18n.changeLanguage('fr') })

  it('renders French labels and the day by default', async () => {
    await i18n.changeLanguage('fr')
    render(<DayStepper day="2026-06-28" onChange={() => {}} />)
    expect(screen.getByRole('button', { name: 'Jour précédent' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Jour suivant' })).toBeInTheDocument()
    expect(screen.getByText('28 juin 2026')).toBeInTheDocument()
  })

  describe('in English', () => {
    beforeAll(async () => { await i18n.changeLanguage('en') })

    it('renders English aria-labels and an Intl-formatted day', () => {
      render(<DayStepper day="2026-06-28" onChange={() => {}} />)
      expect(screen.getByRole('button', { name: 'Previous day' })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Next day' })).toBeInTheDocument()
      expect(screen.getByText('June 28, 2026')).toBeInTheDocument()
    })

    it('calls onChange with the adjacent day when stepping', () => {
      const onChange = vi.fn()
      render(<DayStepper day="2026-06-28" onChange={onChange} />)
      fireEvent.click(screen.getByRole('button', { name: 'Next day' }))
      expect(onChange).toHaveBeenCalledWith('2026-06-29')
      fireEvent.click(screen.getByRole('button', { name: 'Previous day' }))
      expect(onChange).toHaveBeenCalledWith('2026-06-27')
    })

    it('disables the next button at maxDay and enables prev', () => {
      render(<DayStepper day="2026-06-28" onChange={() => {}} maxDay="2026-06-28" />)
      expect(screen.getByRole('button', { name: 'Next day' })).toBeDisabled()
      expect(screen.getByRole('button', { name: 'Previous day' })).not.toBeDisabled()
    })

    it('disables the prev button at minDay and enables next', () => {
      render(<DayStepper day="2026-05-01" onChange={() => {}} minDay="2026-05-01" maxDay="2026-06-30" />)
      expect(screen.getByRole('button', { name: 'Previous day' })).toBeDisabled()
      expect(screen.getByRole('button', { name: 'Next day' })).not.toBeDisabled()
    })

    it('does not call onChange when stepping past a disabled bound', () => {
      const onChange = vi.fn()
      render(<DayStepper day="2026-06-28" onChange={onChange} maxDay="2026-06-28" />)
      fireEvent.click(screen.getByRole('button', { name: 'Next day' }))
      expect(onChange).not.toHaveBeenCalled()
    })
  })
})
