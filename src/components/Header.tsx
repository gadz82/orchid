'use client';

import Link from 'next/link';
import { Search } from 'lucide-react';
import { siteConfig } from '@/site-config';
import OrchidIcon from '@/components/OrchidIcon';
import SearchModal from '@/components/SearchModal';
import { useState, useEffect } from 'react';

const TOP_LINKS = [
  { href: '/quickstart', label: 'Quickstart' },
  { href: '/concepts/agents', label: 'Concepts' },
  { href: '/packages/orchid', label: 'Packages' },
  { href: '/examples', label: 'Examples' },
];

export default function Header() {
  const [searchOpen, setSearchOpen] = useState(false);

  // Global ⌘K / Ctrl+K shortcut
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setSearchOpen((prev) => !prev);
      }
    }
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  return (
    <>
      <header className="sticky top-0 z-40 w-full border-b border-orchid-border bg-orchid-bg/90 backdrop-blur">
        <div className="flex h-14 items-center gap-4 px-6">
          <Link href="/" className="flex items-center gap-2 font-semibold text-orchid-text hover:text-orchid-accent transition-colors">
            <OrchidIcon size={28} />
            <span>Orchestrator Index</span>
          </Link>
          <nav aria-label="Top navigation" className="hidden md:flex items-center gap-1 ml-4">
            {TOP_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="px-3 py-1.5 text-sm text-orchid-muted hover:text-orchid-text transition-colors rounded-md hover:bg-orchid-surface"
              >
                {link.label}
              </Link>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            <a
              href={siteConfig.repoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:block text-sm text-orchid-muted hover:text-orchid-text transition-colors"
              aria-label="GitHub repository"
            >
              GitHub ↗
            </a>
            <button
              aria-label="Search documentation (⌘K)"
              onClick={() => setSearchOpen(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-orchid-border bg-orchid-surface text-orchid-muted text-sm hover:text-orchid-text hover:border-orchid-accent transition-colors"
            >
              <Search size={14} />
              <span className="hidden lg:inline">Search</span>
              <kbd className="hidden lg:inline-flex items-center text-xs opacity-60">⌘K</kbd>
            </button>
          </div>
        </div>
      </header>
      {searchOpen && <SearchModal onClose={() => setSearchOpen(false)} />}
    </>
  );
}
