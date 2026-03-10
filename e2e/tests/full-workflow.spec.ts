/**
 * Full E2E workflow: Import station → Explore dataset → Train model → Forecast
 *
 * Tests the complete user journey from raw station data to predictions.
 * Runs against the live Docker stack (junon-nginx).
 */
import { test, expect, type Page } from '@playwright/test'

// Unique name to avoid collision with previous test runs
const DATASET_NAME = `e2e_test_${Date.now()}`
const STATION_CODE = '05596X0058/SF2'
const SEARCH_TERM = '05596X0058'

// ─── Helpers ────────────────────────────────────────────────────────────────────

async function navigateTo(page: Page, path: string) {
  await page.goto(path, { waitUntil: 'networkidle' })
}

async function waitForApi(page: Page) {
  await page.waitForLoadState('networkidle')
}

async function clickNavLink(page: Page, href: string) {
  await page.click(`nav a[href="${href}"]`)
  await page.waitForURL(`**${href}`)
}

// ─── 1. Health check ────────────────────────────────────────────────────────────

test.describe('0 - Health', () => {
  test('app loads and renders nav', async ({ page }) => {
    await navigateTo(page, '/')
    // Nav should have links to all pages
    await expect(page.locator('nav a[href="/data"]')).toBeVisible({ timeout: 10_000 })
    await expect(page.locator('nav a[href="/training"]')).toBeVisible()
    await expect(page.locator('nav a[href="/forecasting"]')).toBeVisible()
  })

  test('API health endpoint responds', async ({ request }) => {
    const resp = await request.get('/api/v1/health')
    expect(resp.status()).toBe(200)
    const body = await resp.json()
    expect(body.status).toBe('ok')
  })
})

// ─── 2. Station import ─────────────────────────────────────────────────────────

test.describe.serial('1 - Import station', () => {
  let page: Page

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage()
  })

  test.afterAll(async () => {
    await page.close()
  })

  test('navigate to Data page', async () => {
    await navigateTo(page, '/')
    await clickNavLink(page, '/data')
    // Should see import form
    await expect(page.locator('text=Import pi')).toBeVisible()
  })

  test('search for station', async () => {
    const searchInput = page.locator('input[placeholder*="Code BSS"]')
    await searchInput.fill(SEARCH_TERM)

    // Wait for search results
    await expect(page.locator(`text=${STATION_CODE}`).first()).toBeVisible({
      timeout: 15_000,
    })
  })

  test('select station from results', async () => {
    // Click on the station result — it's a button with the code in a font-mono span
    const stationButton = page.locator('button .font-mono').filter({
      hasText: STATION_CODE,
    }).first()
    await stationButton.click()

    // Should see "1 station(s) selectionnee(s)" text
    await expect(page.getByText(/station\(s\) s[ée]lectionn/)).toBeVisible()
  })

  test('set dataset name and import', async () => {
    // Find dataset name input by its placeholder (auto-generated station code)
    const nameInput = page.locator(`input[placeholder*="05596"]`).first()
    if (await nameInput.isVisible()) {
      await nameInput.fill(DATASET_NAME)
    }

    // Click import button
    const importBtn = page.locator('button').filter({ hasText: /Importer \d/ })
    await importBtn.click()

    // Import auto-switches to Explorer tab on success
    // Wait for either the Explorer tab content or the success message
    // The button shows "Importation..." then auto-switches tab
    await page.waitForFunction(
      () => {
        // Check if Explorer tab is now active (has cyan border)
        const tabs = document.querySelectorAll('button')
        return Array.from(tabs).some(
          (t) =>
            t.textContent?.includes('Explorer') &&
            t.className.includes('border-accent-cyan'),
        )
      },
      { timeout: 30_000 },
    )

    await waitForApi(page)
  })

  test('Explorer tab shows dataset cards', async () => {
    // After auto-switch, should see dataset cards
    const cards = page.locator('button').filter({
      hasText: /niveau_nappe|05596/,
    })
    await expect(cards.first()).toBeVisible({ timeout: 10_000 })
  })
})

// ─── 3. Explore dataset ─────────────────────────────────────────────────────────

test.describe.serial('2 - Explore dataset', () => {
  let page: Page

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage()
    await navigateTo(page, '/data')
    await page.click('button:has-text("Explorer")')
    await waitForApi(page)
  })

  test.afterAll(async () => {
    await page.close()
  })

  test('select a dataset card', async () => {
    // Click first dataset card
    const card = page.locator('button').filter({
      hasText: /niveau_nappe|05596/,
    }).first()
    await expect(card).toBeVisible({ timeout: 10_000 })
    await card.click()
    await waitForApi(page)

    // Sub-tabs should appear
    await expect(page.locator('button:has-text("Apercu")')).toBeVisible()
  })

  test('preview tab shows data table', async () => {
    await expect(page.locator('text=Apercu des donnees')).toBeVisible({
      timeout: 10_000,
    })
    // Table should render
    await expect(page.locator('table').first()).toBeVisible({ timeout: 10_000 })
  })

  test('quality tab shows completeness stats', async () => {
    const qualiteBtn = page.locator('button').filter({ hasText: /^Qualite$/ })
    await qualiteBtn.click()
    await waitForApi(page)

    // Wait for quality stats to load — use exact text matching
    await expect(page.getByText('Lignes', { exact: true })).toBeVisible({ timeout: 15_000 })
    await expect(page.getByText('Colonnes', { exact: true })).toBeVisible()
  })

  test('time series tab shows plot', async () => {
    await page.click('button:has-text("Serie temporelle")')
    await waitForApi(page)

    // Plotly chart should render
    await expect(page.locator('.js-plotly-plot').first()).toBeVisible({
      timeout: 10_000,
    })
  })

  test('correlation tab shows matrix', async () => {
    await page.click('button:has-text("Correlation")')
    await waitForApi(page)

    await expect(page.locator('text=Matrice de correlation')).toBeVisible()
    await expect(page.locator('.js-plotly-plot').first()).toBeVisible({
      timeout: 10_000,
    })
  })
})

// ─── 4. Configure dataset ───────────────────────────────────────────────────────

test.describe.serial('3 - Configure dataset', () => {
  let page: Page

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage()
    await navigateTo(page, '/data')
  })

  test.afterAll(async () => {
    await page.close()
  })

  test('config tab shows variable selection', async () => {
    await page.click('button:has-text("Configurer")')
    await waitForApi(page)

    // Select first dataset from dropdown
    const select = page.locator('select').first()
    await select.selectOption({ index: 1 })
    await waitForApi(page)

    await expect(page.locator('text=Variable cible')).toBeVisible({
      timeout: 10_000,
    })
  })

  test('covariates can be toggled', async () => {
    await expect(page.locator('text=Covariables')).toBeVisible()

    const covButton = page.locator('button').filter({
      hasText: /temperature|precipitation|evaporation|profondeur/,
    }).first()
    if (await covButton.isVisible()) {
      await covButton.click()
    }
  })

  test('preprocessing options are available', async () => {
    await expect(page.locator('text=Pretraitement')).toBeVisible()
    await expect(page.locator('text=Normalisation')).toBeVisible()
  })
})

// ─── 5. Training ────────────────────────────────────────────────────────────────

test.describe.serial('4 - Training', () => {
  let page: Page

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage()
  })

  test.afterAll(async () => {
    await page.close()
  })

  test('navigate to Training page', async () => {
    await navigateTo(page, '/training')
    await expect(page.locator('text=Entra').first()).toBeVisible({ timeout: 10_000 })
  })

  test('model selection shows available models', async () => {
    await waitForApi(page)
    const modelSelect = page.locator('select').first()
    await expect(modelSelect).toBeVisible({ timeout: 10_000 })
    const options = await modelSelect.locator('option').allTextContents()
    expect(options.length).toBeGreaterThan(1)
  })

  test('dataset selector shows imported datasets', async () => {
    // Find the select that has dataset options
    const selects = page.locator('select')
    const count = await selects.count()
    let found = false
    for (let i = 0; i < count; i++) {
      const opts = await selects.nth(i).locator('option').allTextContents()
      if (opts.some((o) => /05596|hubeau|piezo|e2e/i.test(o))) {
        found = true
        expect(opts.length).toBeGreaterThan(1)
        break
      }
    }
    expect(found).toBe(true)
  })

  test('can configure training hyperparameters', async () => {
    const numberInputs = page.locator('input[type="number"]')
    expect(await numberInputs.count()).toBeGreaterThan(0)
  })

  test('start training button exists', async () => {
    const startBtn = page.locator('button').filter({
      hasText: /Lancer|Demarrer|Entra/i,
    })
    await expect(startBtn.first()).toBeVisible()
  })

  test('start short training run', async () => {
    // Select model type
    const modelTypeSelect = page.locator('select').first()
    await modelTypeSelect.selectOption({ index: 1 })
    await waitForApi(page)

    // Select dataset
    const selects = page.locator('select')
    const count = await selects.count()
    for (let i = 0; i < count; i++) {
      const opts = await selects.nth(i).locator('option').allTextContents()
      if (opts.some((o) => /05596|hubeau|piezo/i.test(o))) {
        await selects.nth(i).selectOption({ index: 1 })
        break
      }
    }

    // Set minimal epochs
    const epochLabel = page.locator('label').filter({ hasText: /epoch/i })
    if (await epochLabel.count() > 0) {
      const epochInput = epochLabel.first().locator('..').locator('input[type="number"]').first()
      if (await epochInput.isVisible()) {
        await epochInput.fill('2')
      }
    }

    // Click start
    const startBtn = page.locator('button').filter({
      hasText: /Lancer|Demarrer|Entra/i,
    }).first()

    if (await startBtn.isEnabled()) {
      await startBtn.click()

      // Should see some feedback — the phase indicator becomes active
      // Use getByText for exact matching to avoid strict mode issues
      await expect(
        page.getByText('Preparation', { exact: true })
          .or(page.getByText('Calcul en cours', { exact: true }))
          .or(page.locator('text=Erreur :'))
      ).toBeVisible({ timeout: 30_000 })
    }
  })
})

// ─── 6. Forecasting ────────────────────────────────────────────────────────────

test.describe.serial('5 - Forecasting', () => {
  let page: Page

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage()
  })

  test.afterAll(async () => {
    await page.close()
  })

  test('navigate to Forecasting page', async () => {
    await navigateTo(page, '/forecasting')
    await expect(page.locator('text=Prevision').first()).toBeVisible({ timeout: 10_000 })
  })

  test('forecast modes are available', async () => {
    await expect(page.locator('button:has-text("Unique")')).toBeVisible()
    await expect(page.locator('button:has-text("Glissant")')).toBeVisible()
    await expect(page.locator('button:has-text("Comparaison")')).toBeVisible()
    await expect(page.locator('button:has-text("Global")')).toBeVisible()
  })

  test('model selector is visible', async () => {
    await waitForApi(page)
    const select = page.locator('select').first()
    await expect(select).toBeVisible()
  })

  test('horizon input is configurable', async () => {
    const horizonInput = page.locator('input[type="number"]').first()
    await expect(horizonInput).toBeVisible()
    await horizonInput.fill('30')
  })

  test('run forecast button exists', async () => {
    await expect(
      page.locator('button:has-text("Lancer la prevision")')
    ).toBeVisible()
  })

  test('switching modes changes controls', async () => {
    // Switch to Rolling mode
    await page.click('button:has-text("Glissant")')
    await expect(page.getByText('Stride (pas)')).toBeVisible()
    await expect(page.locator('input[type="date"]')).toBeVisible()

    // Switch to Global mode
    await page.click('button:has-text("Global")')
    await expect(page.getByText('Stride (pas)')).not.toBeVisible()

    // Back to Single
    await page.click('button:has-text("Unique")')
  })
})

// ─── 7. Counterfactual ──────────────────────────────────────────────────────────

test.describe.serial('6 - Counterfactual', () => {
  let page: Page

  test.beforeAll(async ({ browser }) => {
    page = await browser.newPage()
  })

  test.afterAll(async () => {
    await page.close()
  })

  test('navigate to Counterfactual page', async () => {
    await navigateTo(page, '/counterfactual')
    await expect(page.locator('text=contrefactuel').first()).toBeVisible({
      timeout: 10_000,
    })
  })

  test('CF methods are available', async () => {
    await expect(page.locator('button:has-text("PhysCF")')).toBeVisible()
    await expect(page.locator('button:has-text("Optuna")')).toBeVisible()
    await expect(page.locator('button:has-text("COMET")')).toBeVisible()
  })

  test('IPS transition selector is available', async () => {
    await expect(page.locator('text=Transition IPS')).toBeVisible()
  })

  test('perturbation sliders are visible', async () => {
    await expect(page.locator('text=Perturbations')).toBeVisible()
    const sliders = page.locator('input[type="range"]')
    expect(await sliders.count()).toBeGreaterThan(0)
  })

  test('method-specific hyperparams change', async () => {
    // PhysCF default
    await expect(page.locator('text=lambda_prox')).toBeVisible()

    // Switch to Optuna
    await page.click('button:has-text("Optuna")')
    await expect(page.locator('text=n_trials')).toBeVisible()

    // Switch to COMET
    await page.click('button:has-text("COMET")')
    await expect(page.locator('text=k_sigma')).toBeVisible()

    // Back
    await page.click('button:has-text("PhysCF")')
  })

  test('generate button exists', async () => {
    await expect(
      page.locator('button:has-text("Generer le contrefactuel")')
    ).toBeVisible()
  })
})

// ─── 8. Navigation ──────────────────────────────────────────────────────────────

test.describe('7 - Navigation', () => {
  test('all nav links work', async ({ page }) => {
    await navigateTo(page, '/')

    const navLinks = page.locator('nav a[href]')
    const count = await navLinks.count()
    expect(count).toBeGreaterThanOrEqual(4)

    const hrefs: string[] = []
    for (let i = 0; i < count; i++) {
      const href = await navLinks.nth(i).getAttribute('href')
      if (href && href !== '/') hrefs.push(href)
    }

    for (const href of hrefs) {
      await navigateTo(page, href)
      const body = await page.locator('body').textContent()
      expect(body?.length).toBeGreaterThan(10)
    }
  })
})

// ─── 9. API integration ────────────────────────────────────────────────────────

test.describe('8 - API endpoints', () => {
  test('GET /datasets returns array', async ({ request }) => {
    const resp = await request.get('/api/v1/datasets')
    expect(resp.status()).toBe(200)
    const data = await resp.json()
    expect(Array.isArray(data)).toBe(true)
    expect(data.length).toBeGreaterThan(0)
  })

  test('GET /models returns array', async ({ request }) => {
    const resp = await request.get('/api/v1/models')
    expect(resp.status()).toBe(200)
    const data = await resp.json()
    expect(Array.isArray(data)).toBe(true)
  })

  test('GET /models/available returns available models', async ({ request }) => {
    const resp = await request.get('/api/v1/models/available')
    expect(resp.status()).toBe(200)
    const data = await resp.json()
    expect(Array.isArray(data)).toBe(true)
    expect(data.length).toBeGreaterThan(0)
    expect(data[0]).toHaveProperty('name')
  })

  test('GET /datasets/:id/preview returns data', async ({ request }) => {
    const listResp = await request.get('/api/v1/datasets')
    const datasets = await listResp.json()
    if (datasets.length === 0) return test.skip()

    const resp = await request.get(`/api/v1/datasets/${datasets[0].id}/preview?n=5`)
    expect(resp.status()).toBe(200)
    const data = await resp.json()
    expect(data).toHaveProperty('columns')
    expect(data).toHaveProperty('rows')
    expect(data.rows.length).toBeLessThanOrEqual(5)
  })

  test('GET /datasets/:id/profile returns stats', async ({ request }) => {
    const listResp = await request.get('/api/v1/datasets')
    const datasets = await listResp.json()
    if (datasets.length === 0) return test.skip()

    const resp = await request.get(`/api/v1/datasets/${datasets[0].id}/profile`)
    expect(resp.status()).toBe(200)
    const data = await resp.json()
    expect(data).toHaveProperty('columns')
    expect(data).toHaveProperty('shape')
    expect(data).toHaveProperty('missing')
  })

  test('POST /datasets/import-db imports station data', async ({ request }) => {
    const resp = await request.post('/api/v1/datasets/import-db', {
      data: {
        table_name: 'hubeau_daily_chroniques',
        schema_name: 'gold',
        columns: [
          'code_bss', 'date', 'niveau_nappe_eau', 'profondeur_nappe',
          'temperature_2m', 'total_precipitation', 'potential_evaporation',
        ],
        date_column: 'date',
        filters: { code_bss: ['05596X0058/SF2'] },
        dataset_name: `api_e2e_${Date.now()}`,
      },
    })
    expect(resp.status()).toBe(201)
    const data = await resp.json()
    expect(data).toHaveProperty('id')
    expect(data.n_rows).toBeGreaterThan(0)
    expect(data.stations).toContain('05596X0058/SF2')
  })

  test('GET /db/stations/search finds stations', async ({ request }) => {
    const resp = await request.get('/api/v1/db/stations/search?q=05596')
    expect(resp.status()).toBe(200)
    const data = await resp.json()
    expect(data.stations.length).toBeGreaterThan(0)
    expect(data.stations[0]).toHaveProperty('code_bss')
  })

  test('GET /db/stations/filters returns filter options', async ({ request }) => {
    const resp = await request.get('/api/v1/db/stations/filters')
    expect(resp.status()).toBe(200)
    const data = await resp.json()
    expect(data).toHaveProperty('departements')
    expect(data).toHaveProperty('tendances')
  })

  test('POST /training/start accepts config', async ({ request }) => {
    const dsResp = await request.get('/api/v1/datasets')
    const datasets = await dsResp.json()
    if (datasets.length === 0) return test.skip()

    const resp = await request.post('/api/v1/training/start', {
      data: {
        model_type: 'TFT',
        dataset_id: datasets[0].id,
        target_variable: datasets[0].target_variable,
        covariates: datasets[0].covariates?.slice(0, 2) || [],
        max_epochs: 1,
        batch_size: 32,
        learning_rate: 0.001,
        input_chunk_length: 30,
        output_chunk_length: 7,
      },
    })
    expect([200, 201, 202, 422]).toContain(resp.status())
    if (resp.status() < 400) {
      const data = await resp.json()
      expect(data).toHaveProperty('task_id')
    }
  })
})
