'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';
import rawSchema from '@/data/config-schema.json';
import rawPractices from '@/data/config-best-practices.json';

// ── Types ────────────────────────────────────────────────────────────────────

export type ConfigEntry = {
  file: 'orchid.yml' | 'agents.yaml';
  path: string;
  type: string;
  required: boolean;
  default: unknown;
  description: string;
  deprecated: boolean;
  examples: string[];
};

type Props = {
  file: 'orchid.yml' | 'agents.yaml';
};

// ── Static data ───────────────────────────────────────────────────────────────

const SCHEMA = rawSchema as ConfigEntry[];
const PRACTICES = rawPractices as Record<string, string>;

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Stable anchor ID for a config entry. */
export function anchorId(file: 'orchid.yml' | 'agents.yaml', path: string): string {
  const prefix = file === 'orchid.yml' ? 'orchid-yml' : 'agents-yaml';
  return `${prefix}__${path}`;
}

/** Deep-link URL to the detailed reference sub-page. */
function refUrl(file: 'orchid.yml' | 'agents.yaml', path: string): string {
  const slug = file === 'orchid.yml' ? 'infrastructure' : 'agents';
  const hash = anchorId(file, path);
  return `/configuration-reference/${slug}#${hash}`;
}

/** Map example file path to /examples/* website route. */
function exampleRoute(path: string): string | null {
  const ROUTES: Record<string, string> = {
    basketball: '/examples/basketball',
    helpdesk: '/examples/helpdesk',
    restaurant: '/examples/restaurant',
    learning: '/examples/learning',
    'mcp-auth': '/examples/mcp-auth',
    'custom-storage': '/examples/custom-storage',
    'rag-strategies': '/examples/rag-strategies',
    'tool-strategies': '/examples/tool-strategies',
    'prompt-customization': '/examples/prompt-customization',
    graph_kb: '/examples/graph-kb',
    wiki: '/examples/wiki',
  };
  // path looks like "/examples/basketball" already (from extractor)
  if (path.startsWith('/examples/')) return path;
  // fall back to directory-name lookup for raw file paths
  const parts = path.split('/');
  if (parts.length >= 2 && parts[0] === 'examples') {
    return ROUTES[parts[1]] ?? null;
  }
  return null;
}

function formatDefault(value: unknown): string {
  if (value === null || value === undefined) return '—';
  if (value === '') return '""';
  if (typeof value === 'boolean') return value.toString();
  if (typeof value === 'number') return value.toString();
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]';
    return JSON.stringify(value);
  }
  if (typeof value === 'object') return '{}';
  return String(value);
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function ConfigTable({ file }: Props) {
  const [filter, setFilter] = useState('');

  const rows = useMemo(() => {
    const all = SCHEMA.filter((e) => e.file === file);
    if (!filter.trim()) return all;
    const q = filter.toLowerCase();
    return all.filter(
      (e) =>
        e.path.toLowerCase().includes(q) ||
        e.description.toLowerCase().includes(q) ||
        e.type.toLowerCase().includes(q),
    );
  }, [file, filter]);

  return (
    <div className="not-prose my-6">
      {/* Filter input */}
      <div className="mb-4">
        <input
          type="search"
          placeholder="Filter by key or description…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Filter configuration keys"
          className="w-full max-w-sm rounded border border-orchid-border bg-orchid-card px-3 py-1.5 text-sm text-orchid-text placeholder:text-orchid-muted focus:outline-none focus:ring-1 focus:ring-orchid-accent"
        />
        {filter && (
          <span className="ml-3 text-xs text-orchid-muted">
            {rows.length} {rows.length === 1 ? 'result' : 'results'}
          </span>
        )}
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-lg border border-orchid-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-orchid-border bg-orchid-card">
              <th className="px-3 py-2 text-left font-semibold text-orchid-text w-[28%]">Key</th>
              <th className="px-3 py-2 text-left font-semibold text-orchid-text w-[12%]">Type</th>
              <th className="px-3 py-2 text-left font-semibold text-orchid-text w-[7%]">Req.</th>
              <th className="px-3 py-2 text-left font-semibold text-orchid-text w-[12%]">Default</th>
              <th className="px-3 py-2 text-left font-semibold text-orchid-text">Description</th>
              <th className="px-3 py-2 text-left font-semibold text-orchid-text w-[14%]">Examples</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-3 py-6 text-center text-orchid-muted text-sm">
                  No keys match &ldquo;{filter}&rdquo;
                </td>
              </tr>
            ) : (
              rows.map((row) => {
                const id = anchorId(file, row.path);
                const practice = PRACTICES[row.path];
                return (
                  <tr
                    key={row.path}
                    id={id}
                    className="border-b border-orchid-border/50 hover:bg-orchid-card/40 transition-colors"
                  >
                    {/* Key */}
                    <td className="px-3 py-2 align-top font-mono text-xs text-orchid-accent-glow break-all">
                      <Link
                        href={refUrl(file, row.path)}
                        className="hover:underline"
                      >
                        {row.deprecated ? <s>{row.path}</s> : row.path}
                      </Link>
                    </td>

                    {/* Type */}
                    <td className="px-3 py-2 align-top font-mono text-xs text-orchid-muted">
                      {row.type}
                    </td>

                    {/* Required */}
                    <td className="px-3 py-2 align-top text-xs">
                      {row.required ? (
                        <span className="text-orange-400 font-semibold">yes</span>
                      ) : (
                        <span className="text-orchid-muted">no</span>
                      )}
                    </td>

                    {/* Default */}
                    <td className="px-3 py-2 align-top font-mono text-xs text-orchid-muted break-all">
                      {formatDefault(row.default)}
                    </td>

                    {/* Description + best-practice note */}
                    <td className="px-3 py-2 align-top text-xs text-orchid-text leading-relaxed">
                      {row.description || <span className="text-orchid-muted/60">—</span>}
                      {practice && (
                        <div className="mt-2 rounded border-l-2 border-green-500/50 bg-green-500/5 px-2 py-1 text-xs text-orchid-muted">
                          <span className="font-semibold text-green-400 mr-1">✓ Best practice:</span>
                          {practice}
                        </div>
                      )}
                    </td>

                    {/* Example badges */}
                    <td className="px-3 py-2 align-top">
                      <div className="flex flex-wrap gap-1">
                        {row.examples.map((ex) => {
                          const route = exampleRoute(ex);
                          if (!route) return null;
                          const label = route.split('/').pop() ?? ex;
                          return (
                            // next/link prepends basePath (/orchid on GH Pages)
                            // so the badge points to /orchid/examples/* not /examples/*.
                            <Link
                              key={ex}
                              href={route}
                              className="inline-block rounded bg-orchid-accent/10 px-1.5 py-0.5 text-xs font-medium text-orchid-accent hover:bg-orchid-accent/20 transition-colors"
                            >
                              {label}
                            </Link>
                          );
                        })}
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
