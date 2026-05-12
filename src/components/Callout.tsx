import type { ReactNode } from 'react';

type Variant = 'info' | 'warning' | 'best-practice';

type Props = {
  variant?: Variant;
  title?: string;
  children: ReactNode;
};

const VARIANT_STYLES: Record<Variant, { border: string; bg: string; icon: string; titleColor: string }> = {
  info: {
    border: 'border-orchid-accent/40',
    bg: 'bg-orchid-accent/5',
    icon: 'ℹ',
    titleColor: 'text-orchid-accent',
  },
  warning: {
    border: 'border-yellow-500/40',
    bg: 'bg-yellow-500/5',
    icon: '⚠',
    titleColor: 'text-yellow-400',
  },
  'best-practice': {
    border: 'border-green-500/40',
    bg: 'bg-green-500/5',
    icon: '✓',
    titleColor: 'text-green-400',
  },
};

export default function Callout({ variant = 'info', title, children }: Props) {
  const styles = VARIANT_STYLES[variant];
  return (
    <div
      role="note"
      className={`my-4 rounded-lg border-l-4 p-4 ${styles.border} ${styles.bg}`}
    >
      {title && (
        <p className={`text-sm font-semibold mb-1 ${styles.titleColor}`}>
          <span aria-hidden="true" className="mr-1.5">{styles.icon}</span>
          {title}
        </p>
      )}
      <div className="text-sm text-orchid-text [&>p]:m-0">{children}</div>
    </div>
  );
}
