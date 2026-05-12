/**
 * generate-sitemap.mjs
 * Generates public/sitemap.xml and public/docs-index.json from nav.ts + MDX frontmatter.
 * Run as part of `npm run build`.
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = join(__dirname, '..');
const CONTENT_DIR = join(ROOT, 'src', 'content');
const PUBLIC_DIR = join(ROOT, 'public');
// Canonical site URL used to build absolute <loc> entries in sitemap.xml.
// Mirrors siteConfig.siteUrl (the file lives in TS so we can't import here).
// Trailing slash is intentionally omitted — paths joined below already start with "/".
const BASE_URL = 'https://gadz82.github.io/orchid';

mkdirSync(PUBLIC_DIR, { recursive: true });

// ── Read nav routes ───────────────────────────────────────────────────────────
// Parse navNodes from nav.ts without importing TS directly
const navSrc = readFileSync(join(ROOT, 'src', 'nav.ts'), 'utf8');
const routeMatches = [...navSrc.matchAll(/href:\s*'([^']+)'/g)];
const routes = routeMatches.map((m) => m[1]);

// ── Frontmatter parser ────────────────────────────────────────────────────────
function parseFrontmatter(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---/);
  if (!match) return {};
  const data = {};
  for (const line of match[1].split('\n')) {
    const kv = line.match(/^(\w[\w-]*):\s*"?([^"#\n]*)"?\s*$/);
    if (kv) data[kv[1]] = kv[2].trim().replace(/^"(.*)"$/, '$1');
    // arrays: tags, sources, related
    const arr = line.match(/^(\w[\w-]*):\s*\[([^\]]*)\]/);
    if (arr) {
      data[arr[1]] = arr[2].split(',').map((s) => s.trim().replace(/^["']|["']$/g, '')).filter(Boolean);
    }
  }
  return data;
}

function slugToContentPath(route) {
  const parts = route === '/' ? [] : route.replace(/^\//, '').split('/');
  if (parts.length === 0) return null; // home page is not MDX
  const direct = join(CONTENT_DIR, ...parts) + '.mdx';
  if (existsSync(direct)) return direct;
  const index = join(CONTENT_DIR, ...parts, 'index.mdx');
  if (existsSync(index)) return index;
  return null;
}

// ── Build docs-index entries ──────────────────────────────────────────────────
const isoDate = new Date().toISOString().split('T')[0];
const docsIndex = [];

for (const route of routes) {
  const filePath = slugToContentPath(route);
  let meta = {};
  if (filePath) {
    const raw = readFileSync(filePath, 'utf8');
    meta = parseFrontmatter(raw);
  }
  docsIndex.push({
    route,
    title: meta.title ?? route,
    description: meta.description ?? '',
    package: meta.package ?? '',
    section: meta.section ?? '',
    tags: meta.tags ?? [],
    sources: meta.sources ?? [],
  });
}

writeFileSync(join(PUBLIC_DIR, 'docs-index.json'), JSON.stringify(docsIndex, null, 2) + '\n');
console.log('✓ public/docs-index.json');

// ── Build sitemap.xml ─────────────────────────────────────────────────────────
const urlEntries = routes
  .map((route) => {
    const loc = `${BASE_URL}${route}`;
    return `  <url>\n    <loc>${loc}</loc>\n    <lastmod>${isoDate}</lastmod>\n    <changefreq>weekly</changefreq>\n    <priority>${route === '/' ? '1.0' : '0.8'}</priority>\n  </url>`;
  })
  .join('\n');

const sitemap = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urlEntries}
</urlset>
`;

writeFileSync(join(PUBLIC_DIR, 'sitemap.xml'), sitemap);
console.log('✓ public/sitemap.xml');
