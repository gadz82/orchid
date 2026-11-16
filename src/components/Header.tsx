'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Search, ChevronDown, Github } from 'lucide-react';
import { siteConfig } from '@/site-config';
import OrchidIcon from '@/components/OrchidIcon';
import SearchModal from '@/components/SearchModal';
import MobileSidebar from '@/components/MobileSidebar';
import { useState, useEffect } from 'react';
import { navNodes } from '@/nav';

interface MegaMenuSection {
  label: string;
  href: string;
  match: string;
  children: { href: string; label: string }[];
}

// Primary sections (mastra/NestJS docs pattern): always visible in the bar,
// each with a compact dropdown; the `match` prefix drives the active state.
const MEGA_MENU: MegaMenuSection[] = [
  {
    label: 'Docs',
    href: '/concepts/agents',
    match: '/concepts',
    children: navNodes.filter((n) => n.group === 'concepts').map((n) => ({ href: n.href, label: n.label })),
  },
  {
    label: 'Packages',
    href: '/packages/orchid',
    match: '/packages',
    children: navNodes.filter((n) => n.group === 'packages').map((n) => ({ href: n.href, label: n.label })),
  },
  {
    label: 'Examples',
    href: '/examples',
    match: '/examples',
    children: navNodes.filter((n) => n.group === 'examples').map((n) => ({ href: n.href, label: n.label })),
  },
  {
    label: 'Configuration',
    href: '/configuration',
    match: '/configuration',
    children: navNodes.filter((n) => n.group === 'configuration').map((n) => ({ href: n.href, label: n.label })),
  },
];

export default function Header() {
  const [searchOpen, setSearchOpen] = useState(false);
  // usePathname() is null without a router context (e.g. component tests).
  // Treat it as "/" so the bar still renders; active states simply stay off.
  const pathname = usePathname() ?? '';

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

  const isActive = (prefix: string) => pathname.startsWith(prefix);

  return (
    <>
      <header className="sticky top-0 z-40 w-full border-b border-orchid-border bg-orchid-bg/90 backdrop-blur">
        <div className="flex h-14 items-center gap-3 px-4 sm:px-6">
          <MobileSidebar />
          <Link href="/" className="flex items-center gap-2 font-semibold text-orchid-text hover:text-orchid-accent transition-colors">
            <OrchidIcon size={28} />
            <span className="hidden sm:inline">Orchestrator Index</span>
          </Link>
          <nav aria-label="Top navigation" className="hidden md:flex items-center gap-0.5 ml-6">
            <Link
              href="/quickstart"
              aria-current={pathname === '/quickstart' ? 'page' : undefined}
              className={[
                'px-3 py-1.5 text-sm transition-colors rounded-md',
                pathname === '/quickstart'
                  ? 'text-orchid-text font-medium bg-orchid-surface'
                  : 'text-orchid-muted hover:text-orchid-text hover:bg-orchid-surface',
              ].join(' ')}
            >
              Quickstart
            </Link>
            {MEGA_MENU.map((section) => (
              <div key={section.label} className="relative group">
                <Link
                  href={section.href}
                  aria-current={isActive(section.match) ? 'true' : undefined}
                  className={[
                    'flex items-center gap-1 px-3 py-1.5 text-sm transition-colors rounded-md',
                    isActive(section.match)
                      ? 'text-orchid-text font-medium bg-orchid-surface'
                      : 'text-orchid-muted hover:text-orchid-text hover:bg-orchid-surface',
                  ].join(' ')}
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
                          className={[
                            'block px-3 py-1.5 text-sm rounded-md transition-colors whitespace-nowrap',
                            pathname === child.href
                              ? 'text-orchid-accent bg-orchid-accent/10'
                              : 'text-orchid-muted hover:text-orchid-text hover:bg-orchid-surface',
                          ].join(' ')}
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
          <div className="ml-auto flex items-center gap-2.5">
            <a
              href={siteConfig.repoUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="hidden sm:flex items-center gap-1.5 text-sm text-orchid-muted hover:text-orchid-text transition-colors px-2 py-1.5 rounded-md hover:bg-orchid-surface"
              aria-label="GitHub repository"
            >
              <Github size={16} />
              <span className="hidden lg:inline">GitHub</span>
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
