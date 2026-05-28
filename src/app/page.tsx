import type { Metadata } from 'next';
import Link from 'next/link';
import { FileCode2, Layers, Cog, Zap, Database, Plug } from 'lucide-react';
import EcosystemSplice from '@/components/EcosystemSplice';
import PillarCard from '@/components/PillarCard';
import CodeBlock from '@/components/CodeBlock';
import OrchidIcon from '@/components/OrchidIcon';
import HeroCanvas from '@/components/HeroCanvas';
import IntelligenceLayer from '@/components/IntelligenceLayer';

export const metadata: Metadata = {
  title: 'Orchestrator Index — Multi-Agent AI Framework',
  description: 'A SOLID, YAML-driven multi-agent framework built on LangGraph.',
};

const PILLARS = [
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
    href: '/concepts/agents',
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
];

const AGENTS_YAML = `version: "1"

defaults:
  llm:
    model: "ollama/llama3.2"
    temperature: 0.2

agents:
  basketball:
    description: >
      NBA basketball expert with player stats,
      team rosters, and head-to-head comparisons.
    tools:
      - get_player_stats
      - compare_players
      - get_team_roster

  psychologist:
    description: >
      Sports psychologist for player motivation,
      mental toughness, and team dynamics.
    tools:
      - assess_motivation
      - suggest_mental_strategy`;

const ORCHID_YAML = `# orchid.yml — Runtime configuration

agents:
  config_path: examples/basketball/agents.yaml

llm:
  model: ollama/llama3.2
  ollama_api_base: http://localhost:11434

auth:
  dev_bypass: true

rag:
  vector_backend: qdrant
  qdrant_url: http://localhost:6333
  embedding_model: ollama/nomic-embed-text

storage:
  class: examples.basketball.storage.sqlite.OrchidSQLiteChatStorage
  dsn: ./chats.db`;

const RUNTIME_TABS = [
  {
    language: 'bash' as const,
    label: 'API',
    code: `curl -X POST http://localhost:8000/chats/my-chat/messages \\
  -H "Content-Type: application/json" \\
  -d '{"content": "Tell me about LeBron James"}'`,
  },
  {
    language: 'bash' as const,
    label: 'CLI',
    code: `orchid chat send "Tell me about LeBron James" \\
  --config examples/basketball/orchid.yml`,
  },
  {
    language: 'json' as const,
    label: 'MCP',
    code: `{
  "tool": "orchid_ask",
  "arguments": {
    "message": "Tell me about LeBron James",
    "chat_id": "my-chat"
  }
}`,
  },
  {
    language: 'python' as const,
    label: 'SDK',
    code: `from orchid_ai import Orchid

orchid = Orchid.from_config("examples/basketball/orchid.yml")
reply = orchid.chat("Tell me about LeBron James")
print(reply.content)`,
  },
];

export default function HomePage() {
  return (
    <div className="w-full">

      {/* ── Full-bleed animated hero ───────────────────────────── */}
      <section
        className="relative overflow-hidden flex items-center justify-center mb-20"
        style={{ minHeight: 'calc(100vh - 3.5rem)' }}
      >
        {/* Fibonacci-spiral canvas animation */}
        <HeroCanvas />

        {/* Radial vignette: darkens the outer ring so the spiral
            fades into the page background, keeping text legible */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              'radial-gradient(ellipse 65% 65% at 50% 50%, transparent 40%, rgba(13,11,17,0.70) 72%, rgba(13,11,17,0.96) 100%)',
          }}
        />

        {/* Soft top + bottom fade to page bg */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 h-32"
          style={{ background: 'linear-gradient(to bottom, #0D0B11 0%, transparent 100%)' }}
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 bottom-0 h-32"
          style={{ background: 'linear-gradient(to top, #0D0B11 0%, transparent 100%)' }}
        />

        {/* Hero text — sits above canvas and vignette */}
        <div className="relative z-10 text-center px-8 max-w-2xl mx-auto flex flex-col items-center">
          <OrchidIcon size={72} />

          <div className="mt-6 inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border border-orchid-border bg-orchid-surface/80 text-orchid-muted backdrop-blur-sm">
            <span className="w-1.5 h-1.5 rounded-full bg-orchid-accent animate-pulse" />
            Proof of Concept · BETA
          </div>

          <h1
            className="mt-5 text-5xl sm:text-5xl lg:text-6xl font-bold tracking-tight"
            style={{
              color: '#D490D7',
              textShadow:
                '0 0 48px rgba(176,106,179,0.65), 0 0 100px rgba(176,106,179,0.22)',
            }}
          >
            Orchestrator Index
          </h1>
          <p className="mt-1 text-md font-light tracking-widest text-orchid-muted/50 uppercase">
            aka Orchid
          </p>

          <p className="mt-4 text-lg sm:text-xl text-orchid-muted max-w-lg leading-relaxed">
            A SOLID, YAML-driven multi-agent framework built on LangGraph.
          </p>

          <div className="mt-8 flex gap-4 justify-center flex-wrap">
            <Link
              href="/quickstart"
              className="inline-flex items-center gap-2 px-7 py-3 rounded-lg bg-orchid-accent text-white font-medium text-sm hover:bg-orchid-accent-hover transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-orchid-accent focus-visible:outline-offset-2"
            >
              Get Started →
            </Link>
            <Link
              href="/ecosystem"
              className="inline-flex items-center gap-2 px-7 py-3 rounded-lg border border-orchid-border bg-orchid-surface/70 text-orchid-text font-medium text-sm backdrop-blur-sm hover:bg-orchid-card hover:border-orchid-accent/40 transition-colors"
            >
              Ecosystem →
            </Link>
          </div>

          {/* Scroll hint */}
          <div className="mt-14 flex flex-col items-center gap-1.5 opacity-40">
            <span className="text-xs text-orchid-muted tracking-widest uppercase">Scroll</span>
            <svg width="16" height="20" viewBox="0 0 16 20" fill="none" aria-hidden="true">
              <path d="M8 1v14M2 10l6 6 6-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-orchid-muted" />
            </svg>
          </div>
        </div>
      </section>

      {/* ── Content below the fold ────────────────────────────── */}
      <div className="max-w-5xl mx-auto">

        {/* ── Intelligence as a Service ───────────────────────── */}
        <section className="mb-16">
          <IntelligenceLayer />
        </section>

        {/* ── Ecosystem splice ────────────────────────────────── */}
        <section className="mb-16 -mx-6 sm:-mx-8 lg:-mx-12">
          <div className="px-6 sm:px-8 lg:px-12">
            <h2 className="text-2xl font-bold text-orchid-text mb-2">The Ecosystem</h2>
            <p className="text-orchid-muted mb-6 text-sm">
              Five separable packages with a one-direction dependency graph.
            </p>
          </div>
          <EcosystemSplice variant="compact" />
        </section>

        {/* ── Six pillars ────────────────────────────────────── */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold text-orchid-text mb-2">Built on six pillars</h2>
          <p className="text-orchid-muted mb-6 text-sm">
            Every design decision stems from these six principles.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {PILLARS.map((p) => (
              <PillarCard key={p.title} {...p} />
            ))}
          </div>
        </section>

        {/* ── What is Orchestrator Index ──────────────────────── */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold text-orchid-text mb-4">What is Orchestrator Index?</h2>
          <div className="text-sm text-orchid-muted leading-relaxed space-y-4">
            <p>
              Orchestrator Index is an open-source, YAML-driven multi-agent AI framework built on
              top of LangGraph. It lets you define autonomous agents, their tools, skills, and
              behavioural prompts entirely in configuration files — no Python or TypeScript
              required for the majority of use cases. Change a single YAML field and swap from
              OpenAI to Anthropic, Google Gemini, Groq, or a local Ollama model without touching
              a line of code.
            </p>
            <p>
              At its core sits <strong className="text-orchid-text">GenericAgent</strong>, a
              composable agent that delegates to specialised collaborators: a SkillDetector for
              intent matching, an MCPDispatcher for external tool calls, and a SkillExecutor for
              step-by-step reasoning. Each agent can be extended with custom Python classes while
              keeping the framework&rsquo;s SOLID architecture intact — new behaviour is added by
              subclassing, never by modifying existing code.
            </p>
            <p>
              Orchestrator Index ships with a five-level hierarchical RAG system
              (root → tenant → user → chat → agent) backed by Qdrant or ChromaDB, giving you
              fine-grained scoping of retrieved context across multi-tenant deployments.
              Documents are parsed once and indexed automatically, so your agents always have
              access to the most relevant knowledge without manual pipeline management.
            </p>
            <p>
              External tool integration is handled through the Model Context Protocol (MCP).
              Orchestrator Index connects to any MCP server with three authentication modes —
              none for local development, passthrough for forwarding user tokens, and full
              OAuth 2.0 discovery for production-grade integrations. Capabilities are cached
              proactively at session start, so the supervisor never blocks on discovery RPCs
              during a conversation.
            </p>
            <p>
              The framework exposes its intelligence layer through three runtimes: a REST API
              (FastAPI) for web and mobile backends, a command-line interface for quick
              prototyping, and an MCP gateway that lets any MCP-capable host LLM — Claude, GPT,
              Gemini, or custom — call your agents as tools. Chat history is persisted via a
              pluggable storage backend (PostgreSQL or SQLite), and every conversation can be
              resumed, shared, or summarised on demand.
            </p>
            <p>
              Whether you are building a customer support bot, a knowledge-base assistant, or a
              multi-agent research pipeline, Orchestrator Index gives you the
              configuration-first abstraction layer to move fast without sacrificing
              extensibility. The entire ecosystem — library, API, CLI, frontend, and MCP
              gateway — is composable, independently deployable, and designed to grow with
              your project.
            </p>
          </div>
        </section>

        {/* ── Two files, three runtimes ─────────── */}
        <section className="mb-16">
          <h2 className="text-2xl font-bold text-orchid-text mb-2">Two files. Four runtimes.</h2>
          <p className="text-orchid-muted mb-6 text-sm">
            <code className="text-orchid-text">orchid.yml</code> defines runtime settings (LLM, RAG, storage).
            <code className="text-orchid-text ml-1">agents.yaml</code> defines agents, tools, and skills.
            Together they power the API, CLI, MCP gateway, and Python SDK.
          </p>
          <div className="flex flex-col lg:flex-row gap-6">
            <div className="flex-1 min-w-0">
              <CodeBlock
                tabs={[
                  { language: 'yaml', label: 'agents.yaml', code: AGENTS_YAML },
                  { language: 'yaml', label: 'orchid.yml', code: ORCHID_YAML },
                ]}
              />
            </div>
            <div className="flex-1 min-w-0">
              <CodeBlock tabs={RUNTIME_TABS} />
            </div>
          </div>
        </section>

        {/* ── Quickstart teaser ───────────────────────────────── */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-orchid-text mb-2">Run it locally</h2>
          <p className="text-orchid-muted mb-4 text-sm">
            One command starts the demo with Ollama-powered models.
          </p>
          <CodeBlock
            tabs={[
              {
                language: 'bash',
                code: 'docker compose -f docker-compose.demo.yml up --build',
              },
            ]}
          />
          <p className="mt-4 text-sm">
            <Link
              href="/quickstart"
              className="text-orchid-accent hover:text-orchid-accent-hover transition-colors"
            >
              Full quickstart →
            </Link>
          </p>
        </section>

      </div>
    </div>
  );
}
