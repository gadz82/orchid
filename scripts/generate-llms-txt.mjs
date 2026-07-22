/**
 * generate-llms-txt.mjs
 * Generates public/llms.txt and public/llms-full.txt from MDX frontmatter + body.
 * Run as part of `npm run build`.
 */

import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = join(__dirname, '..');
const CONTENT_DIR = join(ROOT, 'src', 'content');
const PUBLIC_DIR = join(ROOT, 'public');
// Canonical site URL used to build absolute URLs in llms.txt / llms-full.txt.
// Kept in sync with scripts/generate-sitemap.mjs and siteConfig.siteUrl.
const BASE_URL = 'https://orchestratorindex.com';

mkdirSync(PUBLIC_DIR, { recursive: true });

// ── Frontmatter parser (no external deps) ────────────────────────────────────
function parseFrontmatter(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) return { data: {}, content: raw };
  const yamlBlock = match[1];
  const content = match[2] ?? '';
  const data = {};
  for (const line of yamlBlock.split('\n')) {
    const kv = line.match(/^(\w[\w-]*):\s*"?([^"]*)"?\s*$/);
    if (kv) data[kv[1]] = kv[2].trim();
  }
  return { data, content };
}

// ── Strip JSX tags from MDX body ─────────────────────────────────────────────
function stripJsx(md) {
  return md
    .replace(/<[A-Z][^>]*\/>/g, '')          // self-closing JSX
    .replace(/<[A-Z][^>]*>[\s\S]*?<\/[A-Z][^>]*>/g, '') // JSX blocks
    .replace(/^\s*import\s+.*$/gm, '')        // import statements
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ── Walk content dir ──────────────────────────────────────────────────────────
function walk(dir, base = []) {
  const entries = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      entries.push(...walk(full, [...base, entry]));
    } else if (entry.endsWith('.mdx')) {
      const slug = entry === 'index.mdx' ? base : [...base, entry.replace('.mdx', '')];
      entries.push({ file: full, slug });
    }
  }
  return entries;
}

const files = walk(CONTENT_DIR);

// ── Group by section ──────────────────────────────────────────────────────────
const SECTION_ORDER = ['home', 'concepts', 'packages', 'configuration', 'examples', 'meta'];
const SECTION_LABELS = {
  home: 'Overview',
  concepts: 'Concepts',
  packages: 'Packages',
  configuration: 'Configuration',
  examples: 'Examples',
  meta: 'Reference',
};
// Normalize MDX section values that match nav groups but use different names
const SECTION_NORMALIZE = { ecosystem: 'home', quickstart: 'home' };

const grouped = new Map();
for (const { file, slug } of files) {
  const raw = readFileSync(file, 'utf8');
  const { data, content } = parseFrontmatter(raw);
  const route = '/' + slug.join('/');
  const rawSection = data.section ?? slug[0] ?? 'meta';
  const section = SECTION_NORMALIZE[rawSection] ?? rawSection;
  if (!grouped.has(section)) grouped.set(section, []);
  grouped.get(section).push({
    route,
    title: data.title ?? slug[slug.length - 1],
    description: data.description ?? '',
    content: stripJsx(content),
    order: parseInt(data.order ?? '99', 10),
  });
}

// Sort within each section
for (const pages of grouped.values()) {
  pages.sort((a, b) => a.order - b.order);
}

// ── Build header ──────────────────────────────────────────────────────────────
const isoDate = new Date().toISOString().split('T')[0];
const HEADER = `# Orchid Documentation
Orchid is a generic, platform-agnostic multi-agent AI framework built on LangGraph (Python) and Next.js.
Audience: developers building or deploying multi-agent AI systems.
Each entry below is a documentation page with its URL and description.
Last generated: ${isoDate}

`;

// ── llms.txt ──────────────────────────────────────────────────────────────────
let llmsTxt = HEADER;
for (const sectionKey of SECTION_ORDER) {
  const pages = grouped.get(sectionKey);
  if (!pages?.length) continue;
  llmsTxt += `## ${SECTION_LABELS[sectionKey] ?? sectionKey}\n\n`;
  for (const p of pages) {
    llmsTxt += `${p.title} — ${p.description}\n${BASE_URL}${p.route}\n\n`;
  }
}
// Any sections not in SECTION_ORDER
for (const [key, pages] of grouped) {
  if (SECTION_ORDER.includes(key)) continue;
  llmsTxt += `## ${key}\n\n`;
  for (const p of pages) {
    llmsTxt += `${p.title} — ${p.description}\n${BASE_URL}${p.route}\n\n`;
  }
}

writeFileSync(join(PUBLIC_DIR, 'llms.txt'), llmsTxt.trimEnd() + '\n');
console.log('✓ public/llms.txt');

// ── llms-full.txt ─────────────────────────────────────────────────────────────
let llmsFullTxt = HEADER;
for (const sectionKey of SECTION_ORDER) {
  const pages = grouped.get(sectionKey);
  if (!pages?.length) continue;
  llmsFullTxt += `## ${SECTION_LABELS[sectionKey] ?? sectionKey}\n\n`;
  for (const p of pages) {
    llmsFullTxt += `### ${p.title}\nURL: ${BASE_URL}${p.route}\nDescription: ${p.description}\n\n${p.content}\n\n---\n\n`;
  }
}
for (const [key, pages] of grouped) {
  if (SECTION_ORDER.includes(key)) continue;
  llmsFullTxt += `## ${key}\n\n`;
  for (const p of pages) {
    llmsFullTxt += `### ${p.title}\nURL: ${BASE_URL}${p.route}\nDescription: ${p.description}\n\n${p.content}\n\n---\n\n`;
  }
}

writeFileSync(join(PUBLIC_DIR, 'llms-full.txt'), llmsFullTxt.trimEnd() + '\n');
console.log('✓ public/llms-full.txt');
