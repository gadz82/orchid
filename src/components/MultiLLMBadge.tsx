const PROVIDERS = ['Anthropic', 'OpenAI', 'Ollama', 'Google'] as const;

type Provider = (typeof PROVIDERS)[number];

type Props = {
  providers?: Provider[];
};

export default function MultiLLMBadge({ providers = [...PROVIDERS] }: Props) {
  return (
    <span className="inline-flex flex-wrap gap-1.5 items-center">
      {providers.map((p) => (
        <span
          key={p}
          className="inline-block px-2 py-0.5 rounded-full text-xs font-medium border border-orchid-border bg-orchid-surface text-orchid-muted"
        >
          {p}
        </span>
      ))}
    </span>
  );
}
