export type NavNode = {
  href: string;
  label: string;
  group: string;
  order: number;
};

export const navNodes: NavNode[] = [
  { href: '/', label: 'Home', group: 'home', order: 0 },
  { href: '/ecosystem', label: 'Ecosystem', group: 'home', order: 1 },
  { href: '/quickstart', label: 'Quickstart', group: 'home', order: 2 },

  { href: '/concepts/agents', label: 'Agents', group: 'concepts', order: 10 },
  { href: '/concepts/supervisor', label: 'Supervisor', group: 'concepts', order: 11 },
  { href: '/concepts/rag', label: 'Hierarchical RAG', group: 'concepts', order: 12 },
  { href: '/concepts/mcp', label: 'MCP Integration', group: 'concepts', order: 13 },
  { href: '/concepts/oauth', label: 'OAuth & Auth', group: 'concepts', order: 14 },
  { href: '/concepts/embeddings', label: 'Embeddings', group: 'concepts', order: 15 },
  { href: '/concepts/multi-llm', label: 'Multi-LLM', group: 'concepts', order: 16 },
  { href: '/concepts/persistence', label: 'Persistence', group: 'concepts', order: 17 },
  { href: '/concepts/tool-strategies', label: 'Tool Strategies', group: 'concepts', order: 18 },
  { href: '/concepts/document-parsing', label: 'Document Parsing', group: 'concepts', order: 19 },
  { href: '/concepts/mini-agents', label: 'Mini-Agents', group: 'concepts', order: 20 },
  { href: '/concepts/pollen-bloom', label: 'Pollen + Bloom', group: 'concepts', order: 21 },
  { href: '/concepts/chat-summarization', label: 'Chat Summarization', group: 'concepts', order: 22 },

  { href: '/packages/orchid', label: 'orchid', group: 'packages', order: 20 },
  { href: '/packages/orchid-api', label: 'orchid-api', group: 'packages', order: 21 },
  { href: '/packages/orchid-cli', label: 'orchid-cli', group: 'packages', order: 22 },
  { href: '/packages/orchid-mcp', label: 'orchid-mcp', group: 'packages', order: 23 },
  { href: '/packages/orchid-frontend', label: 'orchid-frontend', group: 'packages', order: 24 },
  { href: '/packages/orchid-storage-postgres', label: 'orchid-storage-postgres', group: 'packages', order: 25 },
  { href: '/packages/orchid-rag-qdrant', label: 'orchid-rag-qdrant', group: 'packages', order: 26 },
  { href: '/packages/orchid-rag-chroma', label: 'orchid-rag-chroma', group: 'packages', order: 27 },
  { href: '/packages/orchid-rag-neo4j', label: 'orchid-rag-neo4j', group: 'packages', order: 28 },

  { href: '/configuration', label: 'Configuration Atlas', group: 'configuration', order: 30 },
  { href: '/configuration-reference', label: 'Configuration Reference', group: 'configuration', order: 31 },
  { href: '/configuration-reference/infrastructure', label: 'Infrastructure', group: 'configuration', order: 32 },
  { href: '/configuration-reference/agents', label: 'Agents', group: 'configuration', order: 33 },

  { href: '/examples', label: 'Examples', group: 'examples', order: 40 },
  { href: '/examples/basketball', label: 'Basketball', group: 'examples', order: 41 },
  { href: '/examples/helpdesk', label: 'Helpdesk', group: 'examples', order: 42 },
  { href: '/examples/restaurant', label: 'Restaurant', group: 'examples', order: 43 },
  { href: '/examples/recipes', label: 'Recipes', group: 'examples', order: 44 },
  { href: '/examples/learning', label: 'Learning', group: 'examples', order: 45 },
  { href: '/examples/education', label: 'Education Studio', group: 'examples', order: 46 },
  { href: '/examples/mcp-auth', label: 'MCP Auth', group: 'examples', order: 47 },
  { href: '/examples/custom-storage', label: 'Custom Storage', group: 'examples', order: 48 },
  { href: '/examples/rag-strategies', label: 'RAG Strategies', group: 'examples', order: 49 },
  { href: '/examples/tool-strategies', label: 'Tool Strategies', group: 'examples', order: 50 },
  { href: '/examples/prompt-customization', label: 'Prompt Customization', group: 'examples', order: 51 },
  { href: '/examples/graph-kb', label: 'Graph KB', group: 'examples', order: 52 },
  { href: '/examples/wiki', label: 'Wiki', group: 'examples', order: 53 },
  { href: '/examples/orchid-experts', label: 'Orchid Experts', group: 'examples', order: 54 },
  { href: '/examples/gallery-curator', label: 'Gallery Curator', group: 'examples', order: 55 },
  { href: '/examples/festival-producer', label: 'Festival Producer', group: 'examples', order: 56 },
  { href: '/examples/architecture-review', label: 'Architecture Review', group: 'examples', order: 57 },
  { href: '/examples/postgres-storage', label: 'Postgres Storage', group: 'examples', order: 58 },

  { href: '/best-practices', label: 'Best Practices', group: 'meta', order: 60 },
  { href: '/glossary', label: 'Glossary', group: 'meta', order: 61 },
  { href: '/contact', label: 'Contact', group: 'meta', order: 62 },
];

export type NavGroup = {
  key: string;
  label: string;
  nodes: NavNode[];
};

export const NAV_GROUP_LABELS: Record<string, string> = {
  home: 'Overview',
  concepts: 'Concepts',
  packages: 'Packages',
  configuration: 'Configuration',
  examples: 'Examples',
  meta: 'Reference',
};

export function groupedNav(): NavGroup[] {
  const groups = new Map<string, NavNode[]>();
  for (const node of navNodes) {
    if (!groups.has(node.group)) groups.set(node.group, []);
    groups.get(node.group)!.push(node);
  }
  return Array.from(groups.entries()).map(([key, nodes]) => ({
    key,
    label: NAV_GROUP_LABELS[key] ?? key,
    nodes: nodes.sort((a, b) => a.order - b.order),
  }));
}
