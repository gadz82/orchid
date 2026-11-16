/**
 * site-url.mjs
 * Single source of truth for the canonical site URL, shared by the build
 * scripts and `src/site-config.ts`. The value lives in `site.config.json` at
 * the repo root because TS modules can't be imported from plain Node scripts.
 *
 * Trailing slash is stripped so callers can join paths that start with "/".
 */

import { readFileSync } from 'fs';
import { join } from 'path';
import { fileURLToPath } from 'url';

const ROOT = join(fileURLToPath(new URL('.', import.meta.url)), '..');
const { siteUrl } = JSON.parse(readFileSync(join(ROOT, 'site.config.json'), 'utf8'));

export const SITE_URL = siteUrl.replace(/\/$/, '');
