import type { ReactNode } from 'react';

/**
 * Extracts the plain text of an arbitrary React node tree (strings, numbers,
 * nested elements). Used to derive stable heading ids without a rehype plugin.
 */
export function nodeText(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === 'boolean') return '';
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(nodeText).join('');
  const maybe = node as { props?: { children?: ReactNode } };
  if (typeof maybe === 'object' && maybe.props) return nodeText(maybe.props?.children);
  return '';
}

/**
 * GitHub-style slug: lowercase, punctuation stripped, spaces → dashes.
 * The AnchorRail reads ids back from the rendered DOM, so server/client stay
 * consistent by construction.
 */
export function slugify(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-');
}

type HeadingProps = {
  as?: 'h1' | 'h2' | 'h3' | 'h4';
  children: ReactNode;
};

/**
 * Server-safe heading that carries an `id` anchor. Wired into MDX as the
 * h2/h3 override so the "On this page" rail can list them.
 */
export default function Heading({ as: Tag = 'h2', children }: HeadingProps) {
  const id = slugify(nodeText(children));
  return (
    <Tag id={id || undefined}>
      {children}
      {id ? (
        <a
          href={`#${id}`}
          aria-label={`Link to ${nodeText(children)}`}
          className="ml-2 opacity-0 transition-opacity focus-visible:opacity-100 [h2:hover>&]:opacity-40 hover:!opacity-100"
        >
          #
        </a>
      ) : null}
    </Tag>
  );
}
