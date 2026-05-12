import Link from 'next/link';
import type { LucideIcon } from 'lucide-react';

type Props = {
  icon: LucideIcon;
  title: string;
  description: string;
  href: string;
};

export default function PillarCard({ icon: Icon, title, description, href }: Props) {
  return (
    <Link
      href={href}
      className="group flex flex-col gap-3 p-5 rounded-xl border border-orchid-border bg-orchid-card hover:border-orchid-accent/40 hover:bg-orchid-surface transition-colors"
    >
      <div className="w-10 h-10 rounded-lg bg-orchid-accent/10 flex items-center justify-center">
        <Icon size={20} className="text-orchid-accent" />
      </div>
      <h3 className="text-sm font-semibold text-orchid-text group-hover:text-orchid-accent transition-colors">
        {title}
      </h3>
      <p className="text-xs text-orchid-muted leading-relaxed">{description}</p>
    </Link>
  );
}
