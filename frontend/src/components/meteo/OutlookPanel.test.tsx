import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import i18n from '@/i18n/config'
import { OutlookPanel } from './OutlookPanel'

beforeAll(() => { i18n.changeLanguage('fr') })

describe('OutlookPanel', () => {
  it('renders the coming-soon state when outlook is null', () => {
    render(<I18nextProvider i18n={i18n}><OutlookPanel outlook={null} /></I18nextProvider>)
    expect(screen.getByText(/Bientôt/)).toBeInTheDocument()
  })
})
