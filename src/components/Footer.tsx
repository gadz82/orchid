import Link from 'next/link';
import ExternalLink from './ExternalLink';
import OrchidIcon from './OrchidIcon';
import { siteConfig } from '@/site-config';

const EXPLORE_LINKS = [
  { href: '/quickstart', label: 'Quickstart' },
  { href: '/concepts/agents', label: 'Concepts' },
  { href: '/packages/orchid', label: 'Packages' },
  { href: '/configuration', label: 'Configuration' },
  { href: '/examples', label: 'Examples' },
];

const REFERENCE_LINKS = [
  { href: '/best-practices', label: 'Best Practices' },
  { href: '/glossary', label: 'Glossary' },
  { href: siteConfig.contactUrl, label: 'Contact' },
];

export default function Footer() {
  return (
    <footer className="border-t border-orchid-border bg-orchid-surface mt-auto">
      <div className="max-w-screen-xl mx-auto px-6 py-10 grid gap-8 sm:grid-cols-2 lg:grid-cols-4 text-sm text-orchid-muted">
        {/* Brand */}
        <div className="space-y-3">
          <div className="flex items-center gap-2 font-semibold text-orchid-text">
            <OrchidIcon size={22} />
            <span>Orchestrator Index</span>
          </div>
          <p className="text-xs leading-relaxed max-w-[26ch]">
            A SOLID, YAML-driven multi-agent framework built on LangGraph. Agents in YAML, tools via MCP, retrieval scoped by design.
          </p>
        </div>
        {/* Explore */}
        <nav aria-label="Footer — explore">
          <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-orchid-muted/80">Explore</p>
          <ul role="list" className="space-y-1.5">
            {EXPLORE_LINKS.map((l) => (
              <li key={l.href}>
                <Link href={l.href} className="hover:text-orchid-text transition-colors">
                  {l.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        {/* Reference */}
        <nav aria-label="Footer — reference">
          <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-orchid-muted/80">Reference</p>
          <ul role="list" className="space-y-1.5">
            {REFERENCE_LINKS.map((l) => (
              <li key={l.href}>
                <Link href={l.href} className="hover:text-orchid-text transition-colors">
                  {l.label}
                </Link>
              </li>
            ))}
            <li>
              <ExternalLink href={`${siteConfig.siteUrl.replace(/\/$/, '')}/llms.txt`} className="hover:text-orchid-text transition-colors">
                llms.txt
              </ExternalLink>
            </li>
          </ul>
        </nav>
        {/* Community */}
        <div>
          <p className="mb-3 text-xs font-semibold uppercase tracking-widest text-orchid-muted/80">Community</p>
          <ul role="list" className="space-y-1.5">
            <li>
              <ExternalLink href={siteConfig.repoUrl} className="hover:text-orchid-text transition-colors">
                GitHub
              </ExternalLink>
            </li>
            <li>
              <ExternalLink href={siteConfig.linkedinUrl} className="hover:text-orchid-text transition-colors">
                LinkedIn
              </ExternalLink>
            </li>
          </ul>
        </div>
      </div>
      <div className="border-t border-orchid-border/60">
        <div className="max-w-screen-xl mx-auto px-6 py-4 flex flex-col sm:flex-row gap-2 items-start sm:items-center justify-between text-xs text-orchid-muted/80">
          <span>
            Licensed under{' '}
            <ExternalLink
              href="https://opensource.org/licenses/MIT"
              className="hover:text-orchid-text transition-colors"
            >
              {siteConfig.license}
            </ExternalLink>
          </span>
          <span>
            Proudly developed using{' '}
            <ExternalLink
              href={siteConfig.sddUrl}
              className="hover:text-orchid-text transition-colors"
            >
              orchid-sdd
            </ExternalLink>
          </span>
          <span>v{siteConfig.version} · Built with Orchid</span>
        </div>
      </div>
    </footer>
  );
}
