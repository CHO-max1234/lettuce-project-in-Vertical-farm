/** Re-capture Figure 13's Bed45 B6 example from the existing English website.
 * Requires Node.js and Playwright; uses only a loopback HTTP server.
 * Captures observation, trend and environmental panels without changing site data.
 * The original manuscript viewport/cropping is unknown; this is a new capture.
 */
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';

const DEFAULT_SITE = 'D:/Googledrive/양상추 분류모델/practice code/파이프라인 시각화 구현 테스트/GPT_260926/web_en/dist';
const DEFAULT_OUTPUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'reproduced');
const VIEWPORT = {width:1440,height:1000};
const getArg = (name,fallback)=>{const i=process.argv.indexOf(name);return i<0?fallback:process.argv[i+1];};
const site = path.resolve(getArg('--site',DEFAULT_SITE));
const outputDir = path.resolve(getArg('--output-dir',DEFAULT_OUTPUT));
const sha = b=>crypto.createHash('sha256').update(b).digest('hex');
let playwright;
try {playwright = await import('playwright');}
catch(error) {
  if(!process.env.FIGURE_PLAYWRIGHT_MODULE) throw new Error('Install playwright, or set FIGURE_PLAYWRIGHT_MODULE to an existing Playwright index.mjs');
  playwright = await import(pathToFileURL(process.env.FIGURE_PLAYWRIGHT_MODULE).href);
}
await fs.mkdir(outputDir,{recursive:true});
const output = path.join(outputDir,'Figure 13.png');
try {await fs.access(output);throw new Error('Output exists: choose another --output-dir');}
catch(error) {if(error.code!=='ENOENT')throw error;}
const sourceFiles=['index.html','app.js','styles.css','bed45_manifest.json'];
const sourceHashes=Object.fromEntries(await Promise.all(sourceFiles.map(async n=>[n,sha(await fs.readFile(path.join(site,n)))])));
const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.png':'image/png','.jpg':'image/jpeg','.webp':'image/webp','.svg':'image/svg+xml'};
const server=http.createServer(async(req,res)=>{
  try {
    const rel=decodeURIComponent(new URL(req.url,'http://localhost').pathname).replace(/^\/+/, '')||'index.html';
    const target=path.resolve(site,rel);
    if(!target.startsWith(site+path.sep)){res.writeHead(403);res.end();return;}
    const bytes=await fs.readFile(target);res.writeHead(200,{'content-type':types[path.extname(target)]||'application/octet-stream'});res.end(bytes);
  } catch {res.writeHead(404);res.end();}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
try {
  browser=await playwright.chromium.launch({headless:true,...(process.env.FIGURE_BROWSER_EXECUTABLE?{executablePath:process.env.FIGURE_BROWSER_EXECUTABLE}:{})});
  const page=await browser.newPage({viewport:VIEWPORT,deviceScaleFactor:1});
  const errors=[];page.on('pageerror',error=>errors.push(String(error)));
  await page.goto(`http://127.0.0.1:${server.address().port}/`,{waitUntil:'networkidle'});
  await page.waitForFunction(()=>document.getElementById('status-live').textContent.includes('loaded'));
  await page.getByRole('button',{name:'May 4, 2026',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('weight-value')?.textContent.includes('138.7'));
  await page.waitForFunction(()=>document.getElementById('main-photo').complete&&document.getElementById('main-photo').naturalWidth>0);
  const state=await page.evaluate(()=>({date:document.getElementById('selected-date-short').textContent,weight:document.getElementById('weight-value').textContent,slot:document.getElementById('plant-photo-title').textContent,result:document.getElementById('result-bottom').textContent,stage:document.getElementById('stage-value')?.textContent}));
  const captures=[];
  for(const selector of ['.explorer-grid','#timeline','#environment']){
    const element=page.locator(selector);
    const bytes=await element.screenshot({animations:'disabled'});
    captures.push({selector,bytes,bounds:await element.boundingBox()});
  }
  if(errors.length)throw new Error(errors.join('\n'));
  const width=Math.max(...captures.map(c=>Math.ceil(c.bounds.width)));
  await page.setViewportSize({width,height:1000});
  await page.setContent('<html><body style="margin:0;background:white">'+captures.map(c=>`<img style="display:block;width:100%;height:auto" src="data:image/png;base64,${c.bytes.toString('base64')}">`).join('')+'</body></html>');
  await page.locator('img').evaluateAll(images=>Promise.all(images.map(image=>image.decode())));
  await page.screenshot({path:output,fullPage:true,animations:'disabled'});
  for(const n of sourceFiles)if(sha(await fs.readFile(path.join(site,n)))!==sourceHashes[n])throw new Error('Source changed: '+n);
  const report={status:'SCREEN_RECAPTURED_ORIGINAL_LAYOUT_UNVERIFIED',site,viewport:VIEWPORT,state,sourceHashes,panels:captures.map(({selector,bounds})=>({selector,bounds})),outputSha256:sha(await fs.readFile(output)),originalViewportKnown:false,pixelIdenticalToCollectedFigure:false,modelInference:false,pageErrors:errors};
  await fs.writeFile(path.join(outputDir,'Figure 13.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
} finally {
  if(browser)await browser.close();
  await new Promise(resolve=>server.close(resolve));
}
