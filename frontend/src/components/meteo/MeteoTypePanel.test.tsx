// frontend/src/components/meteo/MeteoTypePanel.test.tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MeteoTypePanel } from './MeteoTypePanel'

describe('MeteoTypePanel', () => {
  const visible = { piezo: true, hydro: false }

  it('renders the 5 original type rows', () => {
    render(<MeteoTypePanel visible={visible} onToggle={() => {}} />)
    for (const label of ['Piézomètre', 'Source', 'Pluviomètre', 'Station de débit', 'Avec modèle']) {
      expect(screen.getByText(label)).toBeInTheDocument()
    }
  })

  it('disables the rows without data', () => {
    render(<MeteoTypePanel visible={visible} onToggle={() => {}} />)
    expect(screen.getByRole('checkbox', { name: /Source/ })).toBeDisabled()
    expect(screen.getByRole('checkbox', { name: /Pluviomètre/ })).toBeDisabled()
    expect(screen.getByRole('checkbox', { name: /Avec modèle/ })).toBeDisabled()
  })

  it('toggles active layers', () => {
    const onToggle = vi.fn()
    render(<MeteoTypePanel visible={visible} onToggle={onToggle} />)
    fireEvent.click(screen.getByRole('checkbox', { name: /Station de débit/ }))
    expect(onToggle).toHaveBeenCalledWith('hydro')
  })
})
