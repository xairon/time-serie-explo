// Enregistre des clips vidéo (.webm) des fonctionnalités JUNON via Playwright.
// Lancer avec:
//   NODE_PATH=<...node_modules avec playwright 1.60> PW_CHROME=<chrome-1217> \
//   node scripts/record_demo.js <clip>
// Clips: observatory | station | pastas | ai | explain
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE = process.env.JUNON_BASE || 'http://localhost:49513';
const OUT = path.join(__dirname, '..', 'demo-videos');
const VP = { width: 1280, height: 800 };
const clip = process.argv[2] || 'observatory';
// Identifiants lus dans l'environnement (ne jamais committer de secret).
//   export JUNON_ADMIN_EMAIL=... JUNON_ADMIN_PW=...
const ADMIN = { email: process.env.JUNON_ADMIN_EMAIL, pw: process.env.JUNON_ADMIN_PW };
if (!ADMIN.email || !ADMIN.pw) {
  console.error('ERREUR: définir JUNON_ADMIN_EMAIL et JUNON_ADMIN_PW dans l\'environnement.');
  process.exit(1);
}
const RUAN = '03276X0009/P';

const sleep = (p, ms) => p.waitForTimeout(ms);

async function login(page) {
  await page.goto(`${BASE}/login`, { waitUntil: 'networkidle', timeout: 60000 });
  await page.fill('input[type="email"]', ADMIN.email);
  await page.fill('input[type="password"]', ADMIN.pw);
  await page.click('button[type="submit"]');
  await sleep(page, 3000);
}

async function shot(page, name) {
  await page.screenshot({ path: path.join(OUT, `_verify_${name}.png`) });
}

async function run() {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PW_CHROME });
  const ctx = await browser.newContext({ viewport: VP, recordVideo: { dir: OUT, size: VP } });
  // Skip the first-visit onboarding tour everywhere.
  await ctx.addInitScript(() => {
    try {
      localStorage.setItem('junon.tour.completed.v1', '1');
      localStorage.setItem('junon-lang', 'fr'); // UI en français (audience BRGM)
    } catch (e) {}
  });
  const page = await ctx.newPage();
  const needsLogin = ['pastas', 'forecast', 'ai', 'explain'].includes(clip);
  if (needsLogin) await login(page);

  const tryStep = async (label, fn) => { try { await fn(); } catch (e) { console.log(`  [skip ${label}] ${e.message.split('\n')[0]}`); } };

  if (clip === 'observatory') {
    await page.goto(`${BASE}/`, { waitUntil: 'networkidle', timeout: 60000 });
    await sleep(page, 5500);                                   // carte + tuiles

    // 1) Moteur de recherche
    await tryStep('search', async () => {
      const box = page.getByRole('combobox').first();
      await box.click(); await box.fill('Orléans'); await sleep(page, 2500);
      await shot(page, 'obs_search');
      await page.keyboard.press('Escape'); await box.fill(''); await sleep(page, 800);
    });

    // 2) Tiroir : Filtres + Couches (sections accordéon)
    await tryStep('drawer', async () => {
      await page.getByRole('button', { name: 'Ouvrir le panneau' }).click(); await sleep(page, 2000);
      await shot(page, 'obs_drawer');
      await page.getByText('Filtres', { exact: true }).first().click().catch(() => {}); await sleep(page, 2000);
      await shot(page, 'obs_filters');
      await page.getByText('Couches', { exact: true }).first().click().catch(() => {}); await sleep(page, 2500);
      await shot(page, 'obs_layers');
      // refermer le tiroir pour dégager la carte
      await page.getByRole('button', { name: 'Ouvrir le panneau' }).click().catch(() => {});
      await page.keyboard.press('Escape').catch(() => {});
      await sleep(page, 1000);
    });

    // 4) Frise chronologique (sécheresse)
    await tryStep('timeline', async () => {
      await page.getByText('Chronologie historique').click(); await sleep(page, 1200);
      await page.locator('button:has(svg.lucide-play)').first().click();
      await sleep(page, 9000);
      await shot(page, 'observatory_mid');
      await sleep(page, 9000);
    });

  } else if (clip === 'station') {
    await page.goto(`${BASE}/station/piezo/${RUAN}`, { waitUntil: 'networkidle', timeout: 60000 });
    await sleep(page, 4500);
    await page.mouse.wheel(0, 700); await sleep(page, 2500);   // fiche technique + situation
    await page.mouse.wheel(0, 700); await sleep(page, 2500);   // séries temporelles
    for (const tab of ['Mensuel', 'Annuel', 'Journalier']) {
      const b = page.getByRole('button', { name: tab });
      if (await b.count()) { await b.first().click(); await sleep(page, 2500); }
    }
    await page.mouse.wheel(0, 800); await sleep(page, 3000);   // indices sécheresse
    await shot(page, 'station_mid');
  } else if (clip === 'pastas') {
    await page.goto(`${BASE}/pastas/gallery`, { waitUntil: 'networkidle', timeout: 60000 });
    await sleep(page, 4000);
    await page.locator('button:has(svg.lucide-eye)').first().click(); // ouvrir le modèle Ruan
    await sleep(page, 4500);
    await page.mouse.wheel(0, 600); await sleep(page, 3000);  // observé vs simulé
    await page.mouse.wheel(0, 700); await sleep(page, 3500);  // réponse impulsionnelle / contributions
    await shot(page, 'pastas_mid');

  } else if (clip === 'forecast') {
    await page.goto(`${BASE}/ai/forecasting`, { waitUntil: 'networkidle', timeout: 60000 });
    await sleep(page, 3000);
    // Pas de filtre station (le modèle porte sa station) → liste modèles stable.
    // Sélection robuste par texte contenant le MAE, insensible au format du libellé.
    const wanted = process.env.CLIP_MODEL_MAE || '0.2450';
    const modelVal = await page.locator('select').nth(1).evaluate((sel, mae) => {
      const o = [...sel.options].find((o) => o.text.includes(mae));
      return o ? o.value : null;
    }, wanted);
    await page.locator('select').nth(1).selectOption(modelVal);
    await sleep(page, 5000); // chargement des artefacts du modèle
    // la prévision se déclenche au mouvement du curseur (debounce)
    const slider = page.locator('input[type="range"]').first();
    if (await slider.count()) {
      await slider.evaluate((el) => { el.value = Math.round((+el.max || 100) * 0.6); el.dispatchEvent(new Event('input', { bubbles: true })); el.dispatchEvent(new Event('change', { bubbles: true })); });
      await sleep(page, 4000);
    }
    await shot(page, 'forecast_mid');
    await page.mouse.wheel(0, 500); await sleep(page, 3500);

  } else if (clip === 'explain') {
    // Modèle MULTIVARIÉ (covariables météo) → la prévision ET le panneau d'explicabilité
    await page.goto(`${BASE}/ai/forecasting`, { waitUntil: 'networkidle', timeout: 60000 });
    await sleep(page, 3000);
    const mae = process.env.CLIP_MODEL_MAE || '1.2450'; // NHiTS multivarié
    const mv = await page.locator('select').nth(1).evaluate((sel, m) => {
      const o = [...sel.options].find((o) => o.text.includes(m));
      return o ? o.value : null;
    }, mae);
    await page.locator('select').nth(1).selectOption(mv);
    await sleep(page, 5000);
    // 1) s'attarder sur les DRIVERS météo (importance pluie/ETP/température)
    await page.mouse.wheel(0, 1100); await sleep(page, 5000);
    await shot(page, 'explain_drivers');
    // 2) résidus (qualité) puis comportement (ACF / décomposition du signal)
    await page.mouse.wheel(0, 700); await sleep(page, 3500);
    await page.mouse.wheel(0, 700); await sleep(page, 4000);
    await shot(page, 'explain_mid');

  } else if (clip === 'counterfactual') {
    await page.goto(`${BASE}/counterfactual`, { waitUntil: 'networkidle', timeout: 60000 });
    await sleep(page, 4000);
    await shot(page, 'counterfactual_top'); // mapper l'UI avant de scripter le scénario
    await page.mouse.wheel(0, 500); await sleep(page, 3000);

  } else {
    throw new Error('clip inconnu: ' + clip);
  }

  const vpath = await page.video().path();
  await ctx.close();
  await browser.close();
  const dest = path.join(OUT, `${clip}.webm`);
  fs.renameSync(vpath, dest);
  console.log(`OK -> ${dest} (${Math.round(fs.statSync(dest).size / 1024)} KB)`);
}

run().catch((e) => { console.error('ERREUR:', e.message); process.exit(1); });
