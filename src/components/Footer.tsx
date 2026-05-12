import Link from 'next/link';
import ExternalLink from './ExternalLink';
import { siteConfig } from '@/site-config';

export default function Footer() {
  return (
    <footer className="border-t border-orchid-border bg-orchid-surface mt-auto">
      <div className="max-w-screen-xl mx-auto px-6 py-8 flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between text-sm text-orchid-muted">
        <div className="flex flex-wrap gap-x-6 gap-y-2">
          <ExternalLink href={siteConfig.repoUrl} className="hover:text-orchid-text transition-colors">
            GitHub
          </ExternalLink>
          <Link href={siteConfig.contactUrl} className="hover:text-orchid-text transition-colors">
            Contact
          </Link>
          <ExternalLink href={siteConfig.linkedinUrl} className="hover:text-orchid-text transition-colors">
            LinkedIn
          </ExternalLink>
        </div>
        <div className="shrink-0">
          Licensed under{' '}
          <ExternalLink
            href="https://opensource.org/licenses/MIT"
            className="hover:text-orchid-text transition-colors"
          >
            {siteConfig.license}
          </ExternalLink>
          {' '}· v{siteConfig.version} · Built with Orchid
        </div>
      </div>
    </footer>
  );
}
