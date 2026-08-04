'use client';

import Link from 'next/link';
import { Search, ChevronDown } from 'lucide-react';
import { siteConfig } from '@/site-config';
import OrchidIcon from '@/components/OrchidIcon';
import SearchModal from '@/components/SearchModal';
import MobileSidebar from '@/components/MobileSidebar';
import { useState, useEffect } from 'react';
import { navNodes } from '@/nav';

interface MegaMenuSection {
  label: string;
  href: string;
  children: { href: string; label: string }[];
}

const MEGA_MENU: MegaMenuSection[] = [
  {
    label: 'Concepts',
    href: '/concepts/agents',
    children: navNodes
      .filter((n) => n.group === 'concepts')
      .map((n) => ({ href: n.href, label: n.label })),
  },
  {
    label: 'Packages',
    href: '/packages/orchid',
    children: navNodes
      .filter((n) => n.group === 'packages')
      .map((n) => ({ href: n.href, label: n.label })),
  },
  {
    label: 'Examples',
    href: '/examples',
    children: navNodes
      .filter((n) => n.group === 'examples')
      .map((n) => ({ href: n.href, label: n.label })),
  },
];

export default function Header() {
  const [searchOpen, setSearchOpen] = useState(false);

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
        <div className="flex h-14 items-center gap-3 px-4 sm:px-6">
          <MobileSidebar />
          <Link href="/" className="flex items-center gap-2 font-semibold text-orchid-text hover:text-orchid-accent transition-colors">
            <OrchidIcon size={28} />
            <span className="hidden sm:inline">Orchestrator Index</span>
          </Link>
          <nav aria-label="Top navigation" className="hidden md:flex items-center gap-1 ml-4">
            <Link
              href="/quickstart"
              className="px-3 py-1.5 text-sm text-orchid-muted hover:text-orchid-text transition-colors rounded-md hover:bg-orchid-surface"
            >
              Quickstart
            </Link>
            {MEGA_MENU.map((section) => (
              <div key={section.label} className="relative group">
                <Link
                  href={section.href}
                  className="flex items-center gap-1 px-3 py-1.5 text-sm text-orchid-muted hover:text-orchid-text transition-colors rounded-md hover:bg-orchid-surface"
                >
                  {section.label}
                  <ChevronDown size={12} className="opacity-50 group-hover:rotate-180 transition-transform" />
                </Link>
                <div className="absolute left-0 top-full pt-1 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-150 z-50">
                  <div className="bg-orchid-bg border border-orchid-border rounded-lg shadow-lg p-4 min-w-[220px]">
                    <div className="grid gap-y-1">
                      {section.children.map((child) => (
                        <Link
                          key={child.href}
                          href={child.href}
                          className="block px-3 py-1.5 text-sm text-orchid-muted hover:text-orchid-text hover:bg-orchid-surface rounded-md transition-colors whitespace-nowrap"
                        >
                          {child.label}
                        </Link>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
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
