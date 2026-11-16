/**
 * generate-md-pages.mjs
 * Generates one static `<route>.md` file into the export output (`out/`) for
 * every documentation route, converted from the same MDX sources that render
 * the HTML pages. Runs AFTER `next build` (the export wipes `out/`) so LLM
 * spiders can fetch a markdown twin next to every HTML page.
 *
 * The HTML pages advertise the twins with a `llms-canonical` meta tag (see
 * `src/app/[...slug]/page.tsx`); the sitemap lists them next to the HTML
 * routes. Design record: assessment/adr/ADR-001-markdown-twins.md.
 */

import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const ROOT = join(__dirname, '..');
const CONTENT_DIR = join(ROOT, 'src', 'content');
// Overridable so tests can emit into a scratch directory without touching the
// real export output.
const OUT_DIR = process.env.MD_PAGES_OUT_DIR ? join(ROOT, process.env.MD_PAGES_OUT_DIR) : join(ROOT, 'out');

// Canonical site URL — shared with the other build scripts and siteConfig.
import { SITE_URL as BASE_URL } from './site-url.mjs';

// ── Parse packageRepos from site-config.ts (RepoLink needs the URLs) ─────────
function parsePackageRepos() {
  const src = readFileSync(join(ROOT, 'src', 'site-config.ts'), 'utf8');
  const map = {};
  const block = src.match(/packageRepos:\s*\{([\s\S]*?)\}\s*as const/);
  if (block) {
    for (const m of block[1].matchAll(/'([^']+)':\s*'([^']+)'/g)) {
      map[m[1]] = m[2];
    }
  }
  return map;
}
const PACKAGE_REPOS = parsePackageRepos();

// ── Frontmatter parser (same conventions as the sibling generators) ──────────
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

// ── JSX → markdown converters ────────────────────────────────────────────────
// CodePair renders a YAML/Markdown two-file example as one fenced block each.
function convertCodePair(md) {
  return md.replace(
    /<CodePair\b([\s\S]*?)\/>/g,
    (_, props) => {
      const pick = (name) => {
        const m = props.match(new RegExp(`${name}=\\{\`([\\s\\S]*?)\`\\}`));
        return m ? m[1] : '';
      };
      const pickLabel = (name) => {
        const m = props.match(new RegExp(`${name}="([^"]*)"`));
        return m ? m[1] : '';
      };
      const yaml = pick('yaml');
      const mdBody = pick('md');
      const yamlLabel = pickLabel('yamlLabel');
      const mdLabel = pickLabel('mdLabel');
      let out = '';
      if (yaml) out += `\`\`\`yaml${yamlLabel ? ` title="${yamlLabel}"` : ''}\n${yaml.trimEnd()}\n\`\`\`\n\n`;
      if (mdBody) out += `\`\`\`markdown${mdLabel ? ` title="${mdLabel}"` : ''}\n${mdBody.trimEnd()}\n\`\`\`\n\n`;
      return out;
    },
  );
}

function convertCallout(md) {
  return md.replace(
    /<Callout\b([^>]*?)>\n?([\s\S]*?)<\/Callout>/g,
    (_, props, body) => {
      const type = (props.match(/type="([^"]*)"/) || [])[1] ?? 'info';
      const lines = body.trim().split('\n').map((l) => `> ${l}`);
      return `${lines.join('\n')}\n> \n> *${type} note.*\n`;
    },
  );
}

function convertExternalLinks(md) {
  return md.replace(/<ExternalLink\b([^>]*?)>([\s\S]*?)<\/ExternalLink>/g, (_, props, text) => {
    const href = (props.match(/href="([^"]*)"/) || [])[1] ?? '';
    return `[${text.trim()}](${href})`;
  });
}

function convertRepoLink(md) {
  return md.replace(/<RepoLink\b([^>]*?)\/>/g, (_, props) => {
    const pkg = (props.match(/package="([^"]*)"/) || [])[1] ?? '';
    const href = PACKAGE_REPOS[pkg];
    return href ? `View on GitHub: [${pkg}](${href})\n` : '';
  });
}

function convertMultiLLMBadge(md) {
  return md.replace(
    /<MultiLLMBadge\b[^>]*\/>/g,
    () => 'Providers: OpenAI · Anthropic · Google Gemini · Groq · Ollama · any LiteLLM provider\n',
  );
}

function convertConfigTable(md) {
  return md.replace(
    /<ConfigTable\b[^>]*\/>/g,
    () => 'The full filterable configuration reference is rendered on the [Configuration Atlas](/configuration) page.\n',
  );
}

// Components that carry no textual meaning for a markdown reader.
function dropVisualComponents(md) {
  return md.replace(/<EcosystemSplice\b[^>]*\/>/g, '');
}

function convertJsx(md) {
  return [convertCodePair, convertCallout, convertExternalLinks, convertRepoLink, convertMultiLLMBadge, convertConfigTable, dropVisualComponents].reduce(
    (acc, fn) => fn(acc),
    md,
  );
}

function stripMdx(md) {
  return (
    convertJsx(md)
      .replace(/^\s*import\s+.*$/gm, '') // import statements
      .replace(/\{\/\*[\s\S]*?\*\/\}/g, '') // JSX comments
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  );
}

// ── Home page twin: generated overview (home is TSX, not MDX) ────────────────
function buildHomeMarkdown(entries) {
  const sections = {};
  for (const e of entries) {
    if (!e.route) continue;
    const section = e.section ?? 'pages';
    (sections[section] ??= []).push(e);
  }
  let out = '# Orchid Documentation\n\n';
  out += 'Orchid is a generic, platform-agnostic multi-agent AI framework built on LangGraph (Python) and Next.js.\n';
  out += 'Agents are configured in YAML, tools come from MCP servers, retrieval is scoped hierarchically, and every internal judgement can be routed to a dedicated decision model with deterministic fallbacks.\n\n';
  out += 'Every page below is also available as markdown by appending `.md` to its URL.\n\n';
  for (const [section, pages] of Object.entries(sections)) {
    out += `## ${section}\n\n`;
    for (const p of pages.sort((a, b) => a.order - b.order)) {
      out += `- [${p.title}](${BASE_URL}${p.route}.md) — ${p.description}\n`;
    }
    out += '\n';
  }
  return out;
}

// ── Walk content dir ──────────────────────────────────────────────────────────
function walk(dir, base = []) {
  const entries = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      entries.push(...walk(full, [...base, entry]));
    } else if (entry.endsWith('.mdx')) {
      entries.push({ file: full, slug: entry === 'index.mdx' ? base : [...base, entry.replace('.mdx', '')] });
    }
  }
  return entries;
}

const files = walk(CONTENT_DIR);
const entries = files.map(({ file, slug }) => {
  const raw = readFileSync(file, 'utf8');
  const { data, content } = parseFrontmatter(raw);
  return {
    slug,
    route: '/' + slug.join('/'),
    title: data.title ?? slug.at(-1),
    description: data.description ?? '',
    section: data.section ?? slug[0] ?? 'meta',
    order: parseInt(data.order ?? '99', 10),
    body: stripMdx(content),
    data,
  };
});

mkdirSync(OUT_DIR, { recursive: true });
let count = 0;

for (const e of entries) {
  const filePath = e.slug.length === 0 ? join(OUT_DIR, 'index.md') : join(OUT_DIR, ...e.slug) + '.md';
  mkdirSync(dirname(filePath), { recursive: true });

  // Frontmatter as YAML — same fields the HTML page exposes in <head>/meta.
  const fm = [
    '---',
    `title: "${e.title}"`,
    e.description ? `description: "${e.description}"` : null,
    e.data.package ? `package: ${e.data.package}` : null,
    e.data.section ? `section: ${e.data.section}` : null,
    `html_url: ${BASE_URL}${e.route}`,
    `markdown_url: ${BASE_URL}${e.route}.md`,
    '---',
  ]
    .filter(Boolean)
    .join('\n');

  const md = `${fm}\n\n# ${e.title}\n\n${e.body}\n`;
  writeFileSync(filePath, md);
  count += 1;
}

// Home twin is generated (the home page is TSX, not MDX).
writeFileSync(join(OUT_DIR, 'index.md'), buildHomeMarkdown(entries));
count += 1;

console.log(`✓ out: ${count} markdown pages (docs twins + index overview)`);
