// Entry point — one process serving the JSON API + the built SPA. Config via env.
import { buildApp } from './app.js';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
const here = dirname(fileURLToPath(import.meta.url));
const app = buildApp({
  dbPath: process.env.DATA_FILE ?? './data/catalog.json',
  webDist: process.env.WEB_DIST ?? resolve(here, '../../web/dist'),
});
const port = Number(process.env.PORT ?? 8080);
app.listen({ port, host: '0.0.0.0' })
  .then((addr) => console.log(JSON.stringify({ msg: 'listening', addr })))
  .catch((e) => { console.error(e); process.exit(1); });
