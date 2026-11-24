// Renders assets/header/banner.html to assets/header/banner.png (1600 x 360), the banner at the top of every notebook.
//   node tools/render_banner.js        (needs Playwright with Chromium: npm i -D playwright && npx playwright install chromium)
const { chromium } = require('playwright');
const path = require('path'), { pathToFileURL } = require('url');
(async () => {
  const dir = path.join(__dirname, '..', 'assets', 'header');
  const browser = await chromium.launch();
  const page = await (await browser.newContext({ viewport: { width: 1600, height: 360 }, deviceScaleFactor: 1 })).newPage();
  await page.goto(pathToFileURL(path.join(dir, 'banner.html')).href, { waitUntil: 'load' });
  await page.waitForFunction(() => window.READY === true, null, { timeout: 30000 });
  await page.screenshot({ path: path.join(dir, 'banner.png'), clip: { x: 0, y: 0, width: 1600, height: 360 } });
  await browser.close();
  console.log('wrote assets/header/banner.png');
})().catch(e => { console.error(e); process.exit(1); });
