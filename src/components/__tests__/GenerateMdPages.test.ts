import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import { execFileSync } from 'child_process';
import { readFileSync, rmSync, existsSync } from 'fs';
import { join } from 'path';

import siteConfigJson from '../../../site.config.json';

/**
 * Integration test for the markdown-twin generator (ADR-001). Runs the real
 * script against the real src/content tree, emitting into a scratch dir via
 * MD_PAGES_OUT_DIR so the actual export output is never touched.
 */
const OUT_REL = 'out-md-pages-test';
const SITE_URL = siteConfigJson.siteUrl.replace(/\/$/, '');

function run(script: string): void {
  execFileSync('node', [join('scripts', script)], {
    cwd: join(__dirname, '..', '..', '..'),
    env: { ...process.env, MD_PAGES_OUT_DIR: OUT_REL },
    stdio: 'pipe',
  });
}

describe('generate-md-pages.mjs', () => {
  beforeAll(() => run('generate-md-pages.mjs'));
  afterAll(() => rmSync(join(__dirname, '..', '..', '..', OUT_REL), { recursive: true, force: true }));

  it('emits a markdown twin next to every docs route', () => {
    const root = join(__dirname, '..', '..', '..', OUT_REL);
    for (const rel of ['index.md', 'concepts/rag.md', 'concepts/decision-models.md', 'packages/orchid.md', 'glossary.md']) {
      expect(existsSync(join(root, ...rel.split('/'))), rel).toBe(true);
    }
  });

  it('frontmatter carries html and markdown urls', () => {
    const root = join(__dirname, '..', '..', '..', OUT_REL);
    const md = readFileSync(join(root, 'concepts', 'decision-models.md'), 'utf8');
    expect(md).toContain(`markdown_url: ${SITE_URL}/concepts/decision-models.md`);
    expect(md).toContain(`html_url: ${SITE_URL}/concepts/decision-models`);
  });

  it('converts MDX components to markdown and leaves no JSX residue', () => {
    const root = join(__dirname, '..', '..', '..', OUT_REL);
    const rag = readFileSync(join(root, 'concepts', 'rag.md'), 'utf8');
    expect(rag).not.toMatch(/<[A-Z][A-Za-z]*[ />]/); // no JSX tags survive
    expect(rag).toContain('## '); // headings preserved

    const multi = readFileSync(join(root, 'concepts', 'multi-llm.md'), 'utf8');
    expect(multi).toContain('```yaml'); // fenced code survives
    expect(multi).toMatch(/\[LiteLLM documentation[^\]]*\]\(https:\/\/docs\.litellm\.ai\)/); // ExternalLink → markdown link

    const decision = readFileSync(join(root, 'concepts', 'decision-models.md'), 'utf8');
    expect(decision).toContain('> Decision calls are opt-in'); // Callout → blockquote
  });

  it('home twin is a generated overview linking section pages', () => {
    const root = join(__dirname, '..', '..', '..', OUT_REL);
    const home = readFileSync(join(root, 'index.md'), 'utf8');
    expect(home).toContain('# Orchid Documentation');
    expect(home).toContain('/concepts/decision-models.md');
  });
});
