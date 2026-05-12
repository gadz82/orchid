import type { AnchorHTMLAttributes } from 'react';

type Props = AnchorHTMLAttributes<HTMLAnchorElement>;

export default function ExternalLink({ children, href, ...props }: Props) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    >
      {children}
      <span aria-hidden="true" className="ml-0.5 text-orchid-muted">↗</span>
    </a>
  );
}
