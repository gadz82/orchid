'use client';

import { useState } from 'react';
import { Check, Copy } from 'lucide-react';

export type CodeTab = {
  language: 'python' | 'bash' | 'yaml' | 'typescript' | 'json' | 'http' | 'tsx' | 'text';
  label?: string;
  code: string;
  highlightedHtml?: string;
};

type Props = {
  tabs: CodeTab[];
  filename?: string;
  showLineNumbers?: boolean;
  highlight?: string;
  wrap?: boolean;
};

export default function CodeBlock({ tabs, filename, wrap = false }: Props) {
  const [activeIdx, setActiveIdx] = useState(0);
  const [copied, setCopied] = useState(false);

  const active = tabs[activeIdx] ?? tabs[0];
  const isSingle = tabs.length === 1;

  async function handleCopy() {
    await navigator.clipboard.writeText(active.code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  return (
    <div className="rounded-lg overflow-hidden border border-orchid-border my-4 bg-orchid-card shadow-card">
      {filename && (
        <div className="px-4 py-1.5 text-xs text-orchid-muted border-b border-orchid-border bg-orchid-surface font-mono">
          {filename}
        </div>
      )}

      <div className="flex items-center justify-between border-b border-orchid-border bg-orchid-surface">
        <div role="tablist" className="flex">
          {tabs.map((tab, i) => {
            const label = tab.label ?? tab.language;
            const isActive = i === activeIdx;
            return (
              <button
                key={i}
                role="tab"
                aria-selected={isActive}
                aria-disabled={isSingle}
                tabIndex={isSingle ? -1 : 0}
                onClick={() => !isSingle && setActiveIdx(i)}
                className={[
                  'px-4 py-2 text-xs font-mono transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-orchid-accent focus-visible:outline-offset-[-2px]',
                  isActive
                    ? 'text-orchid-accent border-b-2 border-orchid-accent -mb-px'
                    : 'text-orchid-muted hover:text-orchid-text',
                  isSingle ? 'cursor-default' : 'cursor-pointer',
                ].join(' ')}
              >
                {label}
              </button>
            );
          })}
        </div>

        <button
          onClick={handleCopy}
          aria-label={copied ? 'Copied' : 'Copy code'}
          className="mr-2 p-1.5 rounded text-orchid-muted hover:text-orchid-text transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-orchid-accent"
        >
          {copied ? (
            <Check size={14} className="text-orchid-accent" />
          ) : (
            <Copy size={14} />
          )}
        </button>
      </div>

      <div
        className={[
          'relative text-sm font-mono p-4 overflow-x-auto',
          wrap ? 'whitespace-pre-wrap break-all' : 'whitespace-pre',
        ].join(' ')}
      >
        {active.highlightedHtml ? (
          <div
            className="[&_.shiki]:bg-transparent [&_code]:bg-transparent"
            dangerouslySetInnerHTML={{ __html: active.highlightedHtml }}
          />
        ) : (
          <code className={`language-${active.language} text-orchid-text`}>
            {active.code}
          </code>
        )}
      </div>
    </div>
  );
}
