'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { Search, X } from 'lucide-react';

type PagefindResult = {
  url: string;
  meta: { title?: string };
  excerpt: string;
};

type PagefindSearch = {
  results: Array<{
    data: () => Promise<PagefindResult>;
  }>;
};

type PagefindInstance = {
  search: (query: string) => Promise<PagefindSearch>;
};

declare global {
  interface Window {
    pagefind?: PagefindInstance;
  }
}

async function loadPagefind(): Promise<PagefindInstance | null> {
  if (typeof window === 'undefined') return null;
  if (window.pagefind) return window.pagefind;
  try {
    // Use Function constructor to prevent Vite/bundler static import analysis.
    // pagefind.js is only available after `npm run build` generates the index.
    const url = '/_pagefind/pagefind.js';
    const pf = await (new Function('u', 'return import(u)')(url)) as PagefindInstance;
    window.pagefind = pf;
    return pf;
  } catch {
    return null;
  }
}

type ResultItem = {
  url: string;
  title: string;
  excerpt: string;
};

type SearchModalProps = {
  onClose: () => void;
};

export default function SearchModal({ onClose }: SearchModalProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<ResultItem[]>([]);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  // Close on Escape
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        onClose();
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  // Focus input on mount
  useEffect(() => {
    setTimeout(() => inputRef.current?.focus(), 50);
  }, []);

  // Trap focus inside dialog
  useEffect(() => {
    const el = dialogRef.current;
    if (!el) return;
    function onTab(e: KeyboardEvent) {
      if (e.key !== 'Tab') return;
      const focusable = el!.querySelectorAll<HTMLElement>(
        'input, button, a[href], [tabindex]:not([tabindex="-1"])',
      );
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last?.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first?.focus();
        }
      }
    }
    el.addEventListener('keydown', onTab);
    return () => el.removeEventListener('keydown', onTab);
  }, [results]);

  const runSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setResults([]);
      return;
    }
    setLoading(true);
    const pf = await loadPagefind();
    if (!pf) {
      setLoading(false);
      return;
    }
    const search = await pf.search(q);
    const top = search.results.slice(0, 8);
    const resolved = await Promise.all(
      top.map(async (r) => {
        const data = await r.data();
        return {
          url: data.url,
          title: data.meta?.title ?? data.url,
          excerpt: data.excerpt,
        };
      }),
    );
    setResults(resolved);
    setLoading(false);
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => runSearch(query), 200);
    return () => clearTimeout(timer);
  }, [query, runSearch]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[10vh] px-4"
      aria-modal="true"
      role="dialog"
      aria-label="Search documentation"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Modal chrome */}
      <div
        ref={dialogRef}
        className="relative w-full max-w-xl rounded-xl border border-orchid-border bg-orchid-bg shadow-2xl overflow-hidden"
      >
        {/* Search input row */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-orchid-border">
          <Search size={16} className="text-orchid-muted shrink-0" />
          <input
            ref={inputRef}
            type="search"
            placeholder="Search docs…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 bg-transparent text-orchid-text placeholder:text-orchid-muted outline-none text-sm"
            aria-label="Search query"
          />
          <button
            onClick={onClose}
            aria-label="Close search"
            className="text-orchid-muted hover:text-orchid-text transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-[60vh] overflow-y-auto">
          {loading && (
            <p className="px-4 py-6 text-sm text-orchid-muted text-center">Searching…</p>
          )}
          {!loading && query && results.length === 0 && (
            <p className="px-4 py-6 text-sm text-orchid-muted text-center">
              No results for &ldquo;{query}&rdquo;
            </p>
          )}
          {!loading && results.length > 0 && (
            <ul role="listbox" aria-label="Search results">
              {results.map((r) => (
                <li key={r.url} role="option" aria-selected="false">
                  <a
                    href={r.url}
                    onClick={onClose}
                    className="block px-4 py-3 hover:bg-orchid-surface transition-colors border-b border-orchid-border last:border-0"
                  >
                    <p className="text-sm font-medium text-orchid-text">{r.title}</p>
                    <p
                      className="text-xs text-orchid-muted mt-0.5 line-clamp-2"
                      dangerouslySetInnerHTML={{ __html: r.excerpt }}
                    />
                  </a>
                </li>
              ))}
            </ul>
          )}
          {!query && (
            <p className="px-4 py-6 text-sm text-orchid-muted text-center">
              Type to search the documentation…
            </p>
          )}
        </div>

        {/* Footer hint */}
        <div className="flex items-center gap-3 px-4 py-2 border-t border-orchid-border text-xs text-orchid-muted">
          <span><kbd>↵</kbd> to open</span>
          <span><kbd>Esc</kbd> to close</span>
        </div>
      </div>
    </div>
  );
}
