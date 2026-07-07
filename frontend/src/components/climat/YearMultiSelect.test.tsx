import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import i18n from '@/i18n/config'
import { YearMultiSelect } from './YearMultiSelect'
import { MAX_COMPARE_YEARS, MIN_COMPARE_YEARS } from '@/lib/climat-year-select'

describe('YearMultiSelect', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('renders a chip for every drought preset plus any already-selected year', () => {
    render(<YearMultiSelect years={[1976, 2003]} onChange={() => {}} />)
    expect(screen.getByRole('button', { name: '1976' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '1989' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '2003' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '2022' })).toBeInTheDocument()
  })

  it('marks selected chips as pressed and toggles a year off when clicked', () => {
    const onChange = vi.fn()
    render(<YearMultiSelect years={[1976, 1989, 2003]} onChange={onChange} />)
    const chip = screen.getByRole('button', { name: '1989' })
    expect(chip).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(chip)
    expect(onChange).toHaveBeenCalledWith([1976, 2003])
  })

  it('toggles an unselected year on when clicked', () => {
    const onChange = vi.fn()
    render(<YearMultiSelect years={[1976, 1989, 2003]} onChange={onChange} />)
    const chip = screen.getByRole('button', { name: '2022' })
    expect(chip).toHaveAttribute('aria-pressed', 'false')
    fireEvent.click(chip)
    expect(onChange).toHaveBeenCalledWith([1976, 1989, 2003, 2022])
  })

  it('disables selected chips at MIN_COMPARE_YEARS so the last two years cannot be removed', () => {
    expect(MIN_COMPARE_YEARS).toBe(2)
    render(<YearMultiSelect years={[1976, 2003]} onChange={() => {}} />)
    expect(screen.getByRole('button', { name: '1976' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '2003' })).toBeDisabled()
    // Unselected preset chips stay enabled — the bound only blocks removal, not addition.
    expect(screen.getByRole('button', { name: '1989' })).not.toBeDisabled()
  })

  it('disables unselected chips and the add-year select at MAX_COMPARE_YEARS', () => {
    expect(MAX_COMPARE_YEARS).toBe(6)
    // 6 selected years, one preset (1976) left unselected — it should render disabled.
    const years = [1989, 2003, 2018, 2022, 2025, 2026]
    render(<YearMultiSelect years={years} onChange={() => {}} />)
    const unselectedPresetChip = screen.getByRole('button', { name: '1976' })
    expect(unselectedPresetChip).toBeDisabled()
    expect(unselectedPresetChip).toHaveAttribute('aria-pressed', 'false')
    // A selected chip remains enabled (it can still be removed).
    expect(screen.getByRole('button', { name: '2022' })).not.toBeDisabled()
    const select = screen.getByRole('combobox', { name: 'Ajouter une année' })
    expect(select).toBeDisabled()
  })

  it('does not disable chips or the select below MAX_COMPARE_YEARS', () => {
    render(<YearMultiSelect years={[1976, 1989, 2003]} onChange={() => {}} />)
    expect(screen.getByRole('button', { name: '2022' })).not.toBeDisabled()
    expect(screen.getByRole('combobox', { name: 'Ajouter une année' })).not.toBeDisabled()
  })

  it('calls onChange with the new year when a year is added via the select', () => {
    const onChange = vi.fn()
    render(<YearMultiSelect years={[1976, 1989, 2003]} onChange={onChange} />)
    const select = screen.getByRole('combobox', { name: 'Ajouter une année' })
    fireEvent.change(select, { target: { value: '2010' } })
    expect(onChange).toHaveBeenCalledWith([1976, 1989, 2003, 2010])
  })
})
