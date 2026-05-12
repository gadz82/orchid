import type { ReactElement } from 'react';
import { highlight } from '@/lib/shiki';
import CodeBlock, { type CodeTab } from './CodeBlock';

type CodeProps = {
  className?: string;
  children?: string;
};

type PreProps = {
  children?: ReactElement<CodeProps>;
};

export default async function MdxPre({ children }: PreProps) {
  const codeEl = children as ReactElement<CodeProps> | undefined;
  const className = codeEl?.props?.className ?? '';
  const rawLang = className.replace('language-', '') || 'text';
  const code = String(codeEl?.props?.children ?? '').trimEnd();

  let highlightedHtml: string | undefined;
  if (code.trim().length > 0) {
    try {
      highlightedHtml = await highlight(code, rawLang);
    } catch {
      // Fall back to plain code rendering
    }
  }

  const lang = rawLang as CodeTab['language'];

  return (
    <CodeBlock
      tabs={[{ language: lang, code, highlightedHtml }]}
    />
  );
}
