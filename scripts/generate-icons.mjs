#!/usr/bin/env node
/**
 * Generate raster favicon assets from src/app/icon.svg.
 *
 * Emits:
 *   - src/app/apple-icon.png   180x180 (iOS home-screen)
 *   - src/app/favicon.ico      32x32 PNG-in-ICO (root /favicon.ico)
 *
 * Both are picked up automatically by Next.js App Router's file-system
 * metadata conventions (no <link> tags needed in layout.tsx).
 *
 * Re-run after editing icon.svg:
 *   node scripts/generate-icons.mjs
 */
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import sharp from 'sharp';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, '..');
const svgPath = resolve(root, 'src/app/icon.svg');

const svg = await readFile(svgPath);

// 1. apple-icon.png — 180x180 (iOS spec)
const applePng = await sharp(svg, { density: 384 })
  .resize(180, 180)
  .png()
  .toBuffer();
await writeFile(resolve(root, 'src/app/apple-icon.png'), applePng);

// 2. favicon.ico — single 32x32 PNG embedded in an ICO container.
// Modern browsers (and Next.js's serving) accept PNG-inside-ICO since Vista.
const pngBuf = await sharp(svg, { density: 256 })
  .resize(32, 32)
  .png()
  .toBuffer();

// ICONDIR (6 bytes) + ICONDIRENTRY (16 bytes) + PNG payload
const ico = Buffer.alloc(6 + 16 + pngBuf.length);
// ICONDIR
ico.writeUInt16LE(0, 0);            // reserved
ico.writeUInt16LE(1, 2);            // type = 1 (icon)
ico.writeUInt16LE(1, 4);            // image count
// ICONDIRENTRY
ico.writeUInt8(32, 6);              // width  (32; 256 would be encoded as 0)
ico.writeUInt8(32, 7);              // height (32)
ico.writeUInt8(0, 8);               // palette color count
ico.writeUInt8(0, 9);               // reserved
ico.writeUInt16LE(1, 10);           // color planes
ico.writeUInt16LE(32, 12);          // bits per pixel
ico.writeUInt32LE(pngBuf.length, 14); // bytes in resource
ico.writeUInt32LE(22, 18);          // offset to PNG payload
pngBuf.copy(ico, 22);

await writeFile(resolve(root, 'src/app/favicon.ico'), ico);

console.log('✓ Wrote src/app/apple-icon.png (180x180)');
console.log('✓ Wrote src/app/favicon.ico (32x32 PNG-in-ICO)');
