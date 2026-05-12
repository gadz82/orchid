#!/usr/bin/env tsx
/**
 * Pre-build sanity check: every route in nav.ts must resolve to
 * an existing MDX file in src/content/ or the src/app/page.tsx for '/'.
 */

import { existsSync } from 'fs';
import { join } from 'path';
import { navNodes } from '../src/nav';

// Run from orchid-website/ (the package root)
const ROOT = process.cwd();
const CONTENT_DIR = join(ROOT, 'src', 'content');
const APP_PAGE = join(ROOT, 'src', 'app', 'page.tsx');

function hrefToFilePath(href: string): string | null {
  if (href === '/') return APP_PAGE;

  const parts = href.replace(/^\//, '').split('/');

  const direct = join(CONTENT_DIR, ...parts) + '.mdx';
  if (existsSync(direct)) return direct;

  const index = join(CONTENT_DIR, ...parts, 'index.mdx');
  if (existsSync(index)) return index;

  return null;
}

let failed = false;

for (const node of navNodes) {
  const resolved = hrefToFilePath(node.href);
  if (!resolved) {
    console.error(`[check-nav] MISSING: ${node.href} (expected file in src/content${node.href === '/' ? '' : node.href}.mdx)`);
    failed = true;
  }
}

if (failed) {
  console.error('[check-nav] Some nav routes have no corresponding content file. Fix before building.');
  process.exit(1);
} else {
  console.log(`[check-nav] All ${navNodes.length} nav routes resolved. ✓`);
}
