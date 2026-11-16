import { FileCode2, Layers, Cog, Zap, Database, Plug, GitBranch, Bot, HardDrive } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export type Pillar = {
  icon: LucideIcon;
  title: string;
  description: string;
  href: string;
};

/**
 * Home feature-pillar grid ("Everything a production agent needs").
 * Data-only so it stays unit-testable: Next.js forbids arbitrary exports
 * from page files, so the array lives outside src/app/page.tsx.
 */
export const PILLARS: Pillar[] = [
  {
    icon: FileCode2,
    title: 'YAML-first',
    description:
      'Define agents, skills, tools, and prompts entirely in agents.yaml — no code required for most use cases.',
    href: '/concepts/agents',
  },
  {
    icon: Layers,
    title: 'Multi-LLM',
    description:
      'OpenAI, Anthropic, Google Gemini, Groq, Ollama — switch providers by changing a single YAML field.',
    href: '/concepts/multi-llm',
  },
  {
    icon: GitBranch,
    title: 'Decision Models',
    description:
      'Route internal judgements — routing, guardrails, tool choice, model selection — to a dedicated decision LLM with deterministic fallbacks.',
    href: '/concepts/decision-models',
  },
  {
    icon: Cog,
    title: 'Customizable Agents',
    description:
      'Full control over agentic behaviour, tools, skills, and mini-agents. Compose specialised agents via YAML or Python.',
    href: '/concepts/agents',
  },
  {
    icon: Zap,
    title: 'Event-Driven',
    description:
      'Pollen and Bloom enable reactive, event-driven agent workflows — react to signals, not just prompts.',
    href: '/concepts/pollen-bloom',
  },
  {
    icon: Bot,
    title: 'Mini-Agents',
    description:
      'Fork focused sub-agents for a single turn: parallel agentic loops with curated tool subsets, synthesised back into one answer.',
    href: '/concepts/mini-agents',
  },
  {
    icon: Database,
    title: 'Hierarchical RAG',
    description:
      'Five-level scoped retrieval (root→tenant→user→chat→agent) with pluggable query strategies and Qdrant / ChromaDB built-in.',
    href: '/concepts/rag',
  },
  {
    icon: Plug,
    title: 'MCP-native',
    description:
      'Connect any MCP server. Supports none, passthrough, and OAuth auth modes out of the box.',
    href: '/concepts/mcp',
  },
  {
    icon: HardDrive,
    title: 'Durable & Resumable',
    description:
      'Every thread is checkpointed. Crash, pause, or hand off — resume mid-conversation from SQLite, PostgreSQL, or your own backend.',
    href: '/concepts/persistence',
  },
];
