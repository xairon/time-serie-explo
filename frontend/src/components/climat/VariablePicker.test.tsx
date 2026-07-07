import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import i18n from '@/i18n/config'
import { VariablePicker } from './VariablePicker'

describe('VariablePicker', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('renders all 6 climat variables with SPI first', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    const radios = screen.getAllByRole('radio', { name: /SPI|STI|Bilan|Précip|T°|ETP/ })
    expect(radios.length).toBeGreaterThanOrEqual(6)
    expect(screen.getByText('SPI (précipitations)')).toBeInTheDocument()
  })

  it('marks the active variable as checked', () => {
    render(<VariablePicker variable="sti" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByRole('radio', { name: 'STI (température)' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: 'SPI (précipitations)' })).toHaveAttribute('aria-checked', 'false')
  })

  it('calls onVariableChange when a variable button is clicked', () => {
    const onVariableChange = vi.fn()
    render(<VariablePicker variable="spi" onVariableChange={onVariableChange} window={3} onWindowChange={() => {}} />)
    fireEvent.click(screen.getByRole('radio', { name: 'Bilan hydrique' }))
    expect(onVariableChange).toHaveBeenCalledWith('bilan_hydrique')
  })

  it('shows the window selector for SPI/STI', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByText('Fenêtre')).toBeInTheDocument()
    for (const w of ['1', '3', '6', '12']) {
      expect(screen.getByRole('radio', { name: w })).toBeInTheDocument()
    }
  })

  it('hides the window selector for raw variables', () => {
    render(<VariablePicker variable="precipitation" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.queryByText('Fenêtre')).not.toBeInTheDocument()
  })

  it('calls onWindowChange when a window button is clicked', () => {
    const onWindowChange = vi.fn()
    render(<VariablePicker variable="sti" onVariableChange={() => {}} window={3} onWindowChange={onWindowChange} />)
    fireEvent.click(screen.getByRole('radio', { name: '12' }))
    expect(onWindowChange).toHaveBeenCalledWith(12)
  })
})
