const puppeteer = require('puppeteer');

(async () => {
  console.log("Launching headless browser...");
  const browser = await puppeteer.launch({ 
    headless: "new",
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  
  // Set viewport for dashboard view
  await page.setViewport({ width: 1280, height: 800 });
  
  console.log("Navigating to http://127.0.0.1:5000/...");
  await page.goto('http://127.0.0.1:5000/', { waitUntil: 'networkidle2' });
  
  // Wait for diagnostics backend queries to finish loading
  await new Promise(r => setTimeout(r, 3000));
  
  console.log("Capturing Overview tab...");
  await page.screenshot({ path: 'overview_tab.png' });
  
  // Click Diagnostic Tool tab
  console.log("Switching to Inference / Diagnostic Tool tab...");
  await page.click('[data-tab="inference"]');
  await new Promise(r => setTimeout(r, 1500));
  await page.screenshot({ path: 'inference_tab.png' });
  
  // Click Database Browser tab
  console.log("Switching to Database Browser tab...");
  await page.click('[data-tab="browser"]');
  await new Promise(r => setTimeout(r, 3000)); // wait for database samples to load
  await page.screenshot({ path: 'browser_tab.png' });
  
  // Click Training tab
  console.log("Switching to Training tab...");
  await page.click('[data-tab="training"]');
  await new Promise(r => setTimeout(r, 1500));
  await page.screenshot({ path: 'training_tab.png' });
  
  // Click Evaluation tab
  console.log("Switching to Evaluation tab...");
  await page.click('[data-tab="evaluation"]');
  await new Promise(r => setTimeout(r, 3000)); // wait for charts to render
  await page.screenshot({ path: 'evaluation_tab.png' });
  
  console.log("Screenshots captured successfully!");
  await browser.close();
  process.exit(0);
})().catch(err => {
  console.error("Error capturing screenshots:", err);
  process.exit(1);
});
