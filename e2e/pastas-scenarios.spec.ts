import { test, expect } from '@playwright/test'

test.describe('Pastas Scenarios Page', () => {
  test('renders scenarios page with model selector', async ({ page }) => {
    await page.goto('/pastas/scenarios')
    await expect(page.getByText('Pastas — Scenarios')).toBeVisible()
    await expect(page.getByText('Base Model')).toBeVisible()
    await expect(page.getByText('Modifications')).toBeVisible()
    await expect(page.getByText('Add modification')).toBeVisible()
  })

  test('can add and remove a modification', async ({ page }) => {
    await page.goto('/pastas/scenarios')
    await page.getByText('Add modification').click()
    await page.getByText('pumping synthetic').click()
    await expect(page.getByText('Pumping (synthetic)')).toBeVisible()

    const deleteBtn = page.getByRole('button').filter({ has: page.locator('svg') }).last()
    await deleteBtn.click()
  })
})
