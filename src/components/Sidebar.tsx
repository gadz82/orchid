'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { groupedNav } from '@/nav';

export default function Sidebar() {
  const pathname = usePathname();
  const groups = groupedNav();

  return (
    <nav aria-label="Site navigation" className="w-64 shrink-0 hidden lg:block">
      <div className="sticky top-14 h-[calc(100vh-3.5rem)] overflow-y-auto py-6 pr-4">
        {groups.map((group, gi) => (
          <div key={group.key} className={gi > 0 ? 'mt-6 pt-6 border-t border-orchid-border/60' : ''}>
            <p className="mb-2 px-3 text-[11px] font-semibold uppercase tracking-widest text-orchid-muted/80">
              {group.label}
            </p>
            <ul role="list" className="space-y-0.5">
              {group.nodes.map((node) => {
                const isActive = pathname === node.href;
                return (
                  <li key={node.href}>
                    <Link
                      href={node.href}
                      aria-current={isActive ? 'page' : undefined}
                      className={[
                        'block px-3 py-1.5 rounded-md text-sm transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-orchid-accent',
                        isActive
                          ? 'bg-orchid-accent/10 text-orchid-accent font-medium shadow-[inset_2px_0_0_0] shadow-orchid-accent'
                          : 'text-orchid-muted hover:text-orchid-text hover:bg-orchid-surface',
                      ].join(' ')}
                    >
                      {node.label}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>
    </nav>
  );
}
