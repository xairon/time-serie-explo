import { test, expect } from '@playwright/test'

test('Lab routes redirect to home', async ({ page }) => {
  await page.goto('/lab/latent-space')
  await expect(page).toHaveURL('/')

  await page.goto('/lab/counterfactual')
  await expect(page).toHaveURL('/')

  await page.goto('/lab/pumping-detection')
  await expect(page).toHaveURL('/')
})

test('Pastas nav item exists and navigates to /pastas/fit', async ({ page }) => {
  await page.goto('/')
  const pastasLink = page.getByRole('link', { name: 'Pastas' })
  await expect(pastasLink).toBeVisible()
  await pastasLink.click()
  await expect(page).toHaveURL('/pastas/fit')
  await expect(page.getByText('Pastas — Fit')).toBeVisible()
})
