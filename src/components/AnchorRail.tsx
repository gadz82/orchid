'use client';

import { useEffect, useState } from 'react';
import { usePathname } from 'next/navigation';

type TocEntry = {
  id: string;
  text: string;
  level: 2 | 3;
};

/**
 * Right-hand "On this page" rail (NestJS / mastra docs style). Collects the
 * h2/h3 anchors the server-rendered article already carries and tracks the
 * active one with an IntersectionObserver — no heading data passed through
 * props, so it works for every MDX page without a build step.
 */
export default function AnchorRail() {
  const pathname = usePathname();
  const [entries, setEntries] = useState<TocEntry[]>([]);
  const [activeId, setActiveId] = useState<string>('');

  useEffect(() => {
    const article = document.querySelector('main article');
    if (!article) {
      setEntries([]);
      return;
    }
    const headings = Array.from(article.querySelectorAll<HTMLElement>('h2[id], h3[id]'));
    const next: TocEntry[] = headings.map((h) => ({
      id: h.id,
      text: h.textContent?.replace(/#$/, '').trim() ?? '',
      level: (h.tagName === 'H2' ? 2 : 3) as 2 | 3,
    }));
    setEntries(next);

    if (next.length === 0) return;

    // Scroll-spy: the first heading at/above the viewport top wins; a plain
    // scroll listener is enough for a short docs page and avoids observer
    // edge cases with tall code blocks.
    const computeActive = () => {
      const offset = 96; // header height + breathing room
      let current = next[0]?.id ?? '';
      for (const h of headings) {
        if (h.getBoundingClientRect().top - offset <= 0) current = h.id;
      }
      setActiveId(current);
    };
    computeActive();
    window.addEventListener('scroll', computeActive, { passive: true });
    return () => window.removeEventListener('scroll', computeActive);
  }, [pathname]);

  if (entries.length === 0) return null;

  return (
    <nav aria-label="On this page" className="hidden xl:block w-56 shrink-0">
      <div className="sticky top-14 h-[calc(100vh-3.5rem)] overflow-y-auto py-8 pl-6">
        <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-orchid-muted">On this page</p>
        <ul role="list" className="space-y-1.5 border-l border-orchid-border">
          {entries.map((e) => (
            <li key={e.id}>
              <a
                href={`#${e.id}`}
                aria-current={activeId === e.id ? 'location' : undefined}
                className={[
                  'block text-xs leading-snug transition-colors -ml-px border-l py-0.5',
                  e.level === 3 ? 'pl-5' : 'pl-3',
                  activeId === e.id
                    ? 'border-orchid-accent text-orchid-accent'
                    : 'border-transparent text-orchid-muted hover:text-orchid-text',
                ].join(' ')}
              >
                {e.text}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
