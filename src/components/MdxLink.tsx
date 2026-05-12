import Link from 'next/link';
import type { AnchorHTMLAttributes } from 'react';

/**
 * Markdown link renderer used by MDXRemote. Routes same-origin paths through
 * next/link so Next.js prepends `basePath` (e.g. /orchid) under GitHub Pages.
 *
 * Routing rules:
 *   - href starts with "/"  → <Link> (gets basePath + client-side nav)
 *   - href starts with "#"  → plain <a> (in-page anchor, no basePath needed)
 *   - everything else (http/https/mailto/etc.) → plain <a> with target=_blank
 *     for external http(s) URLs so MDX-authored external links open in a new
 *     tab without authors needing to wrap them in <ExternalLink>.
 */
export default function MdxLink({
  href = '',
  children,
  ...rest
}: AnchorHTMLAttributes<HTMLAnchorElement>) {
  if (href.startsWith('/')) {
    return (
      <Link href={href} {...rest}>
        {children}
      </Link>
    );
  }

  const isExternalHttp = /^https?:\/\//i.test(href);
  return (
    <a
      href={href}
      {...(isExternalHttp ? { target: '_blank', rel: 'noopener noreferrer' } : {})}
      {...rest}
    >
      {children}
    </a>
  );
}
