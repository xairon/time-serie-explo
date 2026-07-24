import { describe, it, expect, vi, beforeAll } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import i18n from '@/i18n/config'
import { VariablePicker } from './VariablePicker'

describe('VariablePicker', () => {
  beforeAll(async () => { await i18n.changeLanguage('fr') })

  it('renders only the 4 index variables, SPI first, and no absolute family', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByRole('radio', { name: 'SPI (précipitations)' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'STI (température)' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'SPEI (précip. − ETP)' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Bilan hydrique' })).toBeInTheDocument()
    // La famille « Valeur absolue » a été retirée (doctrine : cartes = indicateurs).
    expect(screen.queryByRole('radio', { name: 'Précipitations' })).not.toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: 'Température' })).not.toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: 'ETP' })).not.toBeInTheDocument()
    // Les journalières restent (domaine absolu fixe, cf. spec §2.1).
    expect(screen.getByRole('radio', { name: 'Tx (max)' })).toBeInTheDocument()
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
    // Window buttons now carry a labelled accessible name (e.g. "12 — Nappe (12 mois)")
    // instead of the bare number, so we match on the leading number.
    for (const w of ['1', '3', '6', '12']) {
      expect(screen.getByRole('radio', { name: new RegExp(`^${w} —`) })).toBeInTheDocument()
    }
  })

  it('hides the window selector for raw variables', () => {
    render(<VariablePicker variable="bilan_hydrique" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.queryByText('Fenêtre')).not.toBeInTheDocument()
  })

  it('calls onWindowChange when a window button is clicked', () => {
    const onWindowChange = vi.fn()
    render(<VariablePicker variable="sti" onVariableChange={() => {}} window={3} onWindowChange={onWindowChange} />)
    fireEvent.click(screen.getByRole('radio', { name: /^12 —/ }))
    expect(onWindowChange).toHaveBeenCalledWith(12)
  })

  it('renders the daily-temp section with Tx/Tn/T moy', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByText('Données journalières')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Tx (max)' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Tn (min)' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'T moy' })).toBeInTheDocument()
  })

  it('marks the active daily-temp variable as checked and hides the window selector', () => {
    render(<VariablePicker variable="tmax" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByRole('radio', { name: 'Tx (max)' })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByRole('radio', { name: 'Tn (min)' })).toHaveAttribute('aria-checked', 'false')
    expect(screen.queryByText('Fenêtre')).not.toBeInTheDocument()
  })

  it('calls onVariableChange when a daily-temp button is clicked', () => {
    const onVariableChange = vi.fn()
    render(<VariablePicker variable="spi" onVariableChange={onVariableChange} window={3} onWindowChange={() => {}} />)
    fireEvent.click(screen.getByRole('radio', { name: 'Tn (min)' }))
    expect(onVariableChange).toHaveBeenCalledWith('tmin')
  })

  it('groupe les variables sous Anomalie, sans famille Absolu', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByText('Anomalie')).toBeInTheDocument()
    expect(screen.queryByText('Valeur absolue')).not.toBeInTheDocument()
  })

  it('labellise les fenêtres SPI (court terme / saisonnier / long terme / nappe)', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByRole('radio', { name: /Court terme/ })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /Nappe/ })).toBeInTheDocument()
  })

  it('propose la pluie parmi les journalières', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByRole('radio', { name: 'Pluie (jour)' })).toBeInTheDocument()
  })

  it('renders SPEI in the Anomalie group', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByRole('radio', { name: 'SPEI (précip. − ETP)' })).toBeInTheDocument()
  })

  it('shows the window selector when SPEI is active', () => {
    render(<VariablePicker variable="spei" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    expect(screen.getByText('Fenêtre')).toBeInTheDocument()
  })

  it('shows an info tooltip next to SPEI explaining the ETP caveat', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    const speiButton = screen.getByRole('radio', { name: 'SPEI (précip. − ETP)' })
    const infoTip = speiButton.nextElementSibling as HTMLElement
    expect(infoTip).not.toBeNull()
    fireEvent.mouseEnter(infoTip)
    expect(screen.getByText(/pas un Penman-Monteith FAO-56/)).toBeInTheDocument()
  })

  it('does not show an info tooltip next to the other anomaly variables', () => {
    render(<VariablePicker variable="spi" onVariableChange={() => {}} window={3} onWindowChange={() => {}} />)
    const spiButton = screen.getByRole('radio', { name: 'SPI (précipitations)' })
    // SPI is directly followed by the STI radio button, not an InfoTip span.
    expect(spiButton.nextElementSibling?.tagName).toBe('BUTTON')
  })
})
