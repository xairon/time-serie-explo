import { test, expect } from '@playwright/test'

test.describe('Pastas Fit Page', () => {
  test('renders fit page with config form', async ({ page }) => {
    await page.goto('/pastas/fit')
    await expect(page.getByText('Pastas — Fit')).toBeVisible()
    await expect(page.getByText('Station & Data')).toBeVisible()
    await expect(page.getByText('Model Configuration')).toBeVisible()
    await expect(page.getByRole('button', { name: /Fit Model/i })).toBeVisible()
  })

  test('fit button is disabled without dataset selection', async ({ page }) => {
    await page.goto('/pastas/fit')
    const fitBtn = page.getByRole('button', { name: /Fit Model/i })
    await expect(fitBtn).toBeDisabled()
  })
})
