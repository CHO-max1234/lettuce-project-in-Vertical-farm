/** Re-capture FigS9's Bed45 B6 example from the existing English website.
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
const VIEWPORT = {width:1200,height:1000};
const DEVICE_SCALE_FACTOR = 4;
// Applied to the capture page only. Source website files and values are untouched.
const PRINT_CSS = `
 .section-wrap{padding-left:16px;padding-right:16px}
 .explorer-grid{grid-template-columns:minmax(0,1fr);gap:12px}
 .date-rail{display:none}
 .camera-panel,.specimen-photo,.result-panel,.chart-card,.metric-card,.qc-card{padding:16px}
 .date-rail{max-height:none;min-height:0;padding:12px 5px}
 .date-chip{font-size:14px;padding:9px 4px}
 .rail-head{padding:4px 3px 12px;letter-spacing:0;font-size:12px}
 .rail-head strong{font-size:14px}
 .rail-foot{font-size:12px;padding:9px 2px}
 .panel-topline{margin-bottom:10px}
 .panel-topline h3{font-size:23px}
 .micro-label{font-size:13px;letter-spacing:.1em}
 .quiet-badge,.status-chip{font-size:13px;white-space:normal}
 .camera-tabs button{font-size:15px;padding:8px 12px}
 #camera-tabs,.photo-note,#slot-list{display:none}
 .photo-switches{display:none}
 .photo-note,.specimen-foot{font-size:14px;line-height:1.4}
 .photo-controls{font-size:13px}.glass-button{font-size:12px}
 .slot-chip{font-size:14px}
 .specimen-grid{grid-template-columns:minmax(240px,.72fr) minmax(0,1.28fr);gap:12px;margin-top:12px}
 .specimen-photo,.result-panel{min-height:0}
 .crop-stage{height:190px}
 .result-main{gap:10px 14px}
 .result-number span,.stage-label span,.harvest-row span,.harvest-date span{font-size:15px}
 .result-number small,.harvest-date small,.stage-probs,.stage-note{font-size:13px;line-height:1.4}
 .result-number strong{font-size:38px;margin-top:10px}
 .harvest-row strong{font-size:23px}.harvest-date strong{font-size:17px}
 .harvest-row,.harvest-date{padding:10px 0 6px}
 .result-bottom{display:none}
 #timeline,#environment{padding-top:20px;padding-bottom:0}
 .section-heading{margin-bottom:12px;gap:10px}
 .section-heading h2{font-size:30px;margin-bottom:6px}
 .section-heading p:not(.eyebrow){font-size:15px;line-height:1.35}
 .section-heading .eyebrow{font-size:13px;margin-bottom:7px}
 .timeline-grid{grid-template-columns:minmax(0,1.6fr) minmax(310px,.85fr);gap:12px}
 .chart-top{display:block}.chart-top h3{font-size:21px}
 .chart-controls{display:none}
 .chart-top p,.chart-controls label,.chart-controls select{font-size:14px}
 .chart-canvas{height:210px;margin-top:10px}
 .chart-canvas .axis-label{font-size:18px}
 .chart-legend{font-size:14px;gap:10px}
 .chart-disclaimer{font-size:13px;line-height:1.3;margin-top:10px}
 .timeline-aside{display:contents}
 .qc-card{grid-column:1/-1}
 .metric-card h3,.qc-card h3{font-size:20px;margin-bottom:10px}
 .metric-card dt,.metric-card dd{font-size:14px;line-height:1.4}
 .metric-card dl>div{padding:10px 0}
 .qc-card #qc-summary,.qc-card #qc-summary>p:first-child{font-size:14px;line-height:1.35}
 .qc-card .qc-tag{font-size:13px}
 .environment-grid{gap:12px}.environment-card{padding:16px}
 .environment-card>span{font-size:19px;line-height:1.3}
 .environment-card>strong{font-size:40px}
 .environment-card svg{height:70px}
 .environment-note{font-size:12px;margin-bottom:6px}
`;
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
const output = path.join(outputDir,'FigS9.png');
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
  const page=await browser.newPage({viewport:VIEWPORT,deviceScaleFactor:DEVICE_SCALE_FACTOR});
  const errors=[];page.on('pageerror',error=>errors.push(String(error)));
  await page.goto(`http://127.0.0.1:${server.address().port}/`,{waitUntil:'networkidle'});
  await page.waitForFunction(()=>document.getElementById('status-live').textContent.includes('loaded'));
  await page.getByRole('button',{name:'May 4, 2026',exact:true}).click();
  await page.waitForFunction(()=>document.getElementById('weight-value')?.textContent.includes('138.7'));
  await page.waitForFunction(()=>document.getElementById('main-photo').complete&&document.getElementById('main-photo').naturalWidth>0);
  await page.addStyleTag({content:PRINT_CSS});
  await page.evaluate(()=>document.fonts.ready);
  const state=await page.evaluate(()=>({date:document.getElementById('selected-date-short').textContent,weight:document.getElementById('weight-value').textContent,slot:document.getElementById('plant-photo-title').textContent,result:document.getElementById('result-bottom').textContent,stage:document.getElementById('stage-value')?.textContent,harvest:document.getElementById('harvest-value').textContent,environment:['env-temp-value','env-vpd-value','env-ec-value'].map(id=>document.getElementById(id)?.textContent)}));
  const captures=[];
  for(const selector of ['.explorer-grid','#timeline','#environment']){
    const element=page.locator(selector);
    const bytes=await element.screenshot({animations:'disabled'});
    captures.push({selector,bytes,bounds:await element.boundingBox()});
  }
  if(errors.length)throw new Error(errors.join('\n'));
  const width=Math.max(...captures.map(c=>Math.ceil(c.bounds.width)));
  await page.setViewportSize({width,height:1000});
  await page.setContent('<html><body style="margin:0;background:white">'+captures.map(c=>`<img style="display:block;width:${c.bounds.width}px;height:auto;margin:0 auto" src="data:image/png;base64,${c.bytes.toString('base64')}">`).join('')+'</body></html>');
  await page.locator('img').evaluateAll(images=>Promise.all(images.map(image=>image.decode())));
  await page.screenshot({path:output,fullPage:true,animations:'disabled'});
  for(const n of sourceFiles)if(sha(await fs.readFile(path.join(site,n)))!==sourceHashes[n])throw new Error('Source changed: '+n);
  const report={status:'SCREEN_RECAPTURED_SOURCE_UNCHANGED',site,viewport:VIEWPORT,deviceScaleFactor:DEVICE_SCALE_FACTOR,printCss:PRINT_CSS,captureLayout:'Selected date and camera are shown in headings. Navigation controls, repeated slot-guide help and duplicate final-result prose are omitted. Source image, selected crop, model outputs, complete growth-QC message/status tags, chart/method note and environmental context remain visible.',state,sourceHashes,panels:captures.map(({selector,bounds})=>({selector,bounds})),outputSha256:sha(await fs.readFile(output)),recommendedPrintWidthMm:190,effectivePpiAt190mm:width*DEVICE_SCALE_FACTOR/(190/25.4),originalViewportKnown:false,pixelIdenticalToCollectedFigure:false,modelInference:false,pageErrors:errors};
  await fs.writeFile(path.join(outputDir,'FigS9.json'),JSON.stringify(report,null,2));
  console.log(JSON.stringify({status:report.status,state,panels:report.panels,outputSha256:report.outputSha256},null,2));
} finally {
  if(browser)await browser.close();
  await new Promise(resolve=>server.close(resolve));
}
