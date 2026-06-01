'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X } from 'lucide-react';
import { groupedNav } from '@/nav';
import { useState, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';

export default function MobileSidebar() {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  const groups = groupedNav();

  const close = useCallback(() => setOpen(false), []);

  // Close on route change
  useEffect(() => {
    close();
  }, [pathname, close]);

  // Lock body scroll when open
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') close();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, close]);

  return (
    <>
      {/* Hamburger button — visible only below lg */}
      <button
        type="button"
        aria-label="Open navigation menu"
        onClick={() => setOpen(true)}
        className="lg:hidden flex items-center justify-center w-9 h-9 rounded-md border border-orchid-border bg-orchid-surface text-orchid-muted hover:text-orchid-text hover:border-orchid-accent transition-colors"
      >
        <Menu size={18} />
      </button>

      {/* Backdrop + drawer — portalled to body to escape header stacking context */}
      {open && createPortal(
        <div className="fixed inset-0 z-50 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/60 backdrop-blur-sm"
            onClick={close}
            aria-hidden="true"
          />

          {/* Drawer */}
          <nav className="fixed inset-y-0 left-0 w-72 max-w-[85vw] bg-orchid-bg border-r border-orchid-border overflow-y-auto">
            <div className="flex items-center justify-between px-4 h-14 border-b border-orchid-border">
              <span className="text-sm font-semibold text-orchid-text">Navigation</span>
              <button
                type="button"
                aria-label="Close navigation menu"
                onClick={close}
                className="flex items-center justify-center w-8 h-8 rounded-md text-orchid-muted hover:text-orchid-text hover:bg-orchid-surface transition-colors"
              >
                <X size={18} />
              </button>
            </div>

            <div className="py-4 px-3">
              {groups.map((group) => (
                <div key={group.key} className="mb-5">
                  <p className="mb-1.5 px-3 text-xs font-semibold uppercase tracking-wider text-orchid-muted">
                    {group.label}
                  </p>
                  <ul role="list">
                    {group.nodes.map((node) => {
                      const isActive = pathname === node.href;
                      return (
                        <li key={node.href}>
                          <Link
                            href={node.href}
                            aria-current={isActive ? 'page' : undefined}
                            className={[
                              'block px-3 py-2 rounded-md text-sm transition-colors',
                              isActive
                                ? 'bg-orchid-accent/10 text-orchid-accent font-medium'
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
        </div>,
        document.body
      )}
    </>
  );
}
