import Link from 'next/link';

type Props = {
  variant?: 'compact' | 'full';
};

type NodeData = {
  id: string;
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  isAccent: boolean;
  kind: 'pip' | 'boilerplate';
  purpose: string;
};

type EdgeData = {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  labelX: number;
  labelY: number;
  len: number;
  cls: string;
};

const NODES: NodeData[] = [
  {
    id: 'orchid',
    cx: 450,
    cy: 260,
    rx: 75,
    ry: 45,
    isAccent: true,
    kind: 'pip',
    purpose: 'Core framework library — ABCs, GenericAgent, LangGraph graph, RAG, persistence',
  },
  {
    id: 'orchid-api',
    cx: 700,
    cy: 200,
    rx: 65,
    ry: 40,
    isAccent: false,
    kind: 'pip',
    purpose: 'FastAPI server — REST and streaming endpoints, auth, admin',
  },
  {
    id: 'orchid-cli',
    cx: 180,
    cy: 340,
    rx: 65,
    ry: 40,
    isAccent: false,
    kind: 'pip',
    purpose: 'CLI tool — interactive chat, config validation, indexing',
  },
  {
    id: 'orchid-frontend',
    cx: 870,
    cy: 95,
    rx: 65,
    ry: 40,
    isAccent: false,
    kind: 'boilerplate',
    purpose: 'Next.js 15 multi-chat UI — forkable starting point',
  },
  {
    id: 'orchid-mcp',
    cx: 870,
    cy: 415,
    rx: 65,
    ry: 40,
    isAccent: false,
    kind: 'boilerplate',
    purpose: 'MCP gateway — exposes Orchid to any MCP-capable host LLM',
  },
];

const EDGES: EdgeData[] = [
  {
    id: 'e1',
    x1: 635,
    y1: 213,
    x2: 525,
    y2: 240,
    label: 'Python dep',
    labelX: 570,
    labelY: 215,
    len: 113,
    cls: 'eco-e1',
  },
  {
    id: 'e2',
    x1: 245,
    y1: 318,
    x2: 375,
    y2: 278,
    label: 'Python dep',
    labelX: 300,
    labelY: 295,
    len: 136,
    cls: 'eco-e2',
  },
  {
    id: 'e3',
    x1: 808,
    y1: 135,
    x2: 764,
    y2: 160,
    label: 'HTTP',
    labelX: 845,
    labelY: 145,
    len: 55,
    cls: 'eco-e3',
  },
  {
    id: 'e4',
    x1: 836,
    y1: 375,
    x2: 734,
    y2: 240,
    label: 'HTTP',
    labelX: 843,
    labelY: 308,
    len: 160,
    cls: 'eco-e4',
  },
];

export default function EcosystemSplice({ variant = 'full' }: Props) {
  return (
    <div>
      <svg
        viewBox="0 0 1000 520"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Orchid package dependency graph"
        style={{ width: '100%', height: 'auto' }}
      >
        <style>{`
          @keyframes eco-draw { to { stroke-dashoffset: 0; } }
          @keyframes eco-pulse {
            0%, 100% { filter: drop-shadow(0 0 6px rgba(176, 106, 179, 0.25)); }
            50% { filter: drop-shadow(0 0 16px rgba(176, 106, 179, 0.55)); }
          }
          .eco-edge { animation: eco-draw 1.2s ease-out forwards; }
          .eco-e1 { animation-delay: 0s; }
          .eco-e2 { animation-delay: 0.25s; }
          .eco-e3 { animation-delay: 0.5s; }
          .eco-e4 { animation-delay: 0.75s; }
          .eco-center { animation: eco-pulse 3s ease-in-out infinite; }
          @media (prefers-reduced-motion: reduce) {
            .eco-edge { animation: none; stroke-dashoffset: 0; }
            .eco-center { animation: none; }
          }
        `}</style>

        <defs>
          <marker id="eco-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#B06AB3" />
          </marker>
          <filter id="eco-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Edges */}
        {EDGES.map((edge) => (
          <line
            key={edge.id}
            data-testid="edge"
            x1={edge.x1}
            y1={edge.y1}
            x2={edge.x2}
            y2={edge.y2}
            stroke="#B06AB3"
            strokeWidth={1.5}
            strokeOpacity={0.7}
            markerEnd="url(#eco-arrow)"
            className={`eco-edge ${edge.cls}`}
            strokeDasharray={edge.len}
            strokeDashoffset={edge.len}
          />
        ))}

        {/* Edge labels (full variant only) */}
        {variant === 'full' &&
          EDGES.map((edge) => (
            <text
              key={`label-${edge.id}`}
              x={edge.labelX}
              y={edge.labelY}
              textAnchor="middle"
              fontSize={10}
              fill="#7E7394"
              fontFamily="Inter, sans-serif"
            >
              {edge.label}
            </text>
          ))}

        {/* Nodes */}
        {NODES.map((node) => (
          <g
            key={node.id}
            data-testid="node"
            className={node.isAccent ? 'eco-center' : undefined}
          >
            <rect
              x={node.cx - node.rx}
              y={node.cy - node.ry}
              width={node.rx * 2}
              height={node.ry * 2}
              rx={10}
              fill="#1E1A2C"
              stroke={node.isAccent ? '#B06AB3' : '#2A2440'}
              strokeWidth={node.isAccent ? 2 : 1}
            />
            {variant === 'compact' && (
              <text
                x={node.cx}
                y={node.cy}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={11}
                fontWeight="600"
                fill={node.isAccent ? '#D490D7' : '#E8E0F0'}
                fontFamily="Inter, sans-serif"
              >
                {node.id}
              </text>
            )}
            {variant === 'full' && (
              <>
                <rect
                  x={node.cx - 40}
                  y={node.cy + 8}
                  width={80}
                  height={16}
                  rx={8}
                  fill={node.kind === 'pip' ? '#3d1f42' : '#1f1f3d'}
                />
                <text
                  x={node.cx}
                  y={node.cy + 16}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize={9}
                  fill={node.kind === 'pip' ? '#D490D7' : '#7E7394'}
                  fontFamily="Inter, sans-serif"
                >
                  {node.kind === 'pip' ? 'pip package' : 'boilerplate'}
                </text>
              </>
            )}
          </g>
        ))}
      </svg>

      {/* Card grid (full variant only) */}
      {variant === 'full' && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-8">
          {NODES.map((node) => (
            <Link
              key={node.id}
              href={`/packages/${node.id}`}
              className="flex flex-col gap-2 p-4 rounded-xl border border-orchid-border bg-orchid-card hover:border-orchid-accent/40 hover:bg-orchid-surface transition-colors"
            >
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-orchid-text">{node.id}</span>
                <span
                  className="px-2 py-0.5 rounded-full text-xs font-medium"
                  style={{
                    background: node.kind === 'pip' ? '#3d1f42' : '#1f1f3d',
                    color: node.kind === 'pip' ? '#D490D7' : '#7E7394',
                  }}
                >
                  {node.kind === 'pip' ? 'pip package' : 'boilerplate'}
                </span>
              </div>
              <p className="text-xs text-orchid-muted leading-relaxed">{node.purpose}</p>
              <span className="text-xs text-orchid-accent mt-auto">Read docs →</span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
