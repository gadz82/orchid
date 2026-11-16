import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import { execFileSync } from 'child_process';
import { readFileSync, rmSync } from 'fs';
import { join } from 'path';

import siteConfigJson from '../../../site.config.json';

/**
 * Integration test for the sitemap / docs-index generator. Runs the real
 * script against the real src tree, emitting into a scratch dir via
 * SITE_ARTIFACTS_OUT_DIR so the committed artifacts are never touched.
 */
const OUT_REL = 'out-sitemap-test';
const SITE_URL = siteConfigJson.siteUrl.replace(/\/$/, '');
const ROOT = join(__dirname, '..', '..', '..');

function run(): void {
  execFileSync('node', [join('scripts', 'generate-sitemap.mjs')], {
    cwd: ROOT,
    env: { ...process.env, SITE_ARTIFACTS_OUT_DIR: OUT_REL },
    stdio: 'pipe',
  });
}

describe('generate-sitemap.mjs', () => {
  beforeAll(() => run());
  afterAll(() => rmSync(join(ROOT, OUT_REL), { recursive: true, force: true }));

  it('lists the home markdown twin as /index.md, never /.md', () => {
    const xml = readFileSync(join(ROOT, OUT_REL, 'sitemap.xml'), 'utf8');
    expect(xml).toContain(`<loc>${SITE_URL}/index.md</loc>`);
    expect(xml).not.toContain('/.md</loc>');
  });

  it('builds every entry from the canonical site url', () => {
    const xml = readFileSync(join(ROOT, OUT_REL, 'sitemap.xml'), 'utf8');
    const locs = [...xml.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
    expect(locs.length).toBeGreaterThan(0);
    expect(locs.filter((loc) => !loc.startsWith(SITE_URL))).toEqual([]);
  });

  it('exposes the home markdown twin in docs-index', () => {
    const index = JSON.parse(readFileSync(join(ROOT, OUT_REL, 'docs-index.json'), 'utf8')) as Array<{
      route: string;
      markdown: string;
    }>;
    expect(index.find((entry) => entry.route === '/')?.markdown).toBe('/index.md');
  });
});
