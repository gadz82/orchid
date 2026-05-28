'use client';

const SPOKES = [
  { label: 'MCP Gateway', angle: -32, desc: 'Connect any host' },
  { label: 'REST API', angle: 0, desc: 'Programmatic access' },
  { label: 'Python SDK', angle: 32, desc: 'Embed in code' },
] as const;

const HUB_X = 120;
const HUB_Y = 110;
const HUB_R = 36;
const NODE_R = 26;
const SPOKE_LEN = 140;

function spokeEnd(angleDeg: number) {
  const rad = (angleDeg * Math.PI) / 180;
  return {
    x: HUB_X + HUB_R + SPOKE_LEN + Math.cos(rad) * NODE_R,
    y: HUB_Y + Math.sin(rad) * (HUB_R + SPOKE_LEN),
  };
}

function OrchidFlower({ cx, cy, r }: { cx: number; cy: number; r: number }) {
  const petal = (angle: number, color: string) => {
    const rad = (angle * Math.PI) / 180;
    const px = cx + Math.cos(rad) * r * 0.55;
    const py = cy + Math.sin(rad) * r * 0.55;
    return (
      <ellipse
        key={angle}
        cx={px}
        cy={py}
        rx={r * 0.38}
        ry={r * 0.22}
        transform={`rotate(${angle} ${px} ${py})`}
        fill={color}
        opacity={0.9}
      />
    );
  };

  return (
    <g>
      {petal(-90, '#D490D7')}
      {petal(-30, '#C87ECB')}
      {petal(30, '#B06AB3')}
      {petal(90, '#C87ECB')}
      {petal(150, '#B06AB3')}
      {petal(210, '#9A5A9D')}
      <circle cx={cx} cy={cy} r={r * 0.15} fill="#fff" opacity={0.9} />
    </g>
  );
}

export default function IntelligenceLayer() {
  return (
    <div className="flex flex-col lg:flex-row items-center gap-8 lg:gap-12">
      {/* SVG hub-and-spoke diagram */}
      <svg
        viewBox="0 -20 520 260"
        className="w-full max-w-lg flex-shrink-0"
        aria-label="Orchestrator Index creates the intelligence layer — exposed via MCP, API, or SDK"
      >
        <defs>
          <radialGradient id="hubGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#B06AB3" stopOpacity={0.4} />
            <stop offset="100%" stopColor="#B06AB3" stopOpacity={0} />
          </radialGradient>
          <filter id="nodeGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Hub glow */}
        <circle cx={HUB_X} cy={HUB_Y} r={HUB_R * 2.5} fill="url(#hubGlow)">
          <animate attributeName="r" values={`${HUB_R * 2.2};${HUB_R * 2.8};${HUB_R * 2.2}`} dur="4s" repeatCount="indefinite" />
        </circle>

        {/* Spokes */}
        {SPOKES.map((spoke) => {
          const end = spokeEnd(spoke.angle);
          return (
            <g key={spoke.label}>
              {/* Static track */}
              <line
                x1={HUB_X + HUB_R}
                y1={HUB_Y}
                x2={end.x}
                y2={end.y}
                stroke="#2A2440"
                strokeWidth={2}
              />
              {/* Animated dash */}
              <line
                x1={HUB_X + HUB_R}
                y1={HUB_Y}
                x2={end.x}
                y2={end.y}
                stroke="#B06AB3"
                strokeWidth={1.5}
                strokeDasharray="6 8"
                opacity={0.5}
              >
                <animate
                  attributeName="stroke-dashoffset"
                  from="0"
                  to="-28"
                  dur="1.8s"
                  repeatCount="indefinite"
                />
              </line>
              {/* Flowing particle */}
              <circle r={3} fill="#D490D7" opacity={0.9}>
                <animateMotion
                  dur="2.2s"
                  repeatCount="indefinite"
                  path={`M ${HUB_X + HUB_R} ${HUB_Y} L ${end.x} ${end.y}`}
                />
              </circle>
              <circle r={2} fill="#C87ECB" opacity={0.5}>
                <animateMotion
                  dur="2.2s"
                  repeatCount="indefinite"
                  begin="0.7s"
                  path={`M ${HUB_X + HUB_R} ${HUB_Y} L ${end.x} ${end.y}`}
                />
              </circle>
            </g>
          );
        })}

        {/* Node circles */}
        {SPOKES.map((spoke) => {
          const end = spokeEnd(spoke.angle);
          return (
            <g key={`node-${spoke.label}`} filter="url(#nodeGlow)">
              <circle
                cx={end.x}
                cy={end.y}
                r={NODE_R}
                fill="#1E1A2C"
                stroke="#B06AB3"
                strokeWidth={1.5}
              />
              <text
                x={end.x}
                y={end.y + 1}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={10}
                fontWeight={600}
                fill="#E8E0F0"
                fontFamily="Inter, sans-serif"
              >
                {spoke.label}
              </text>
            </g>
          );
        })}

        {/* Hub circle */}
        <circle
          cx={HUB_X}
          cy={HUB_Y}
          r={HUB_R}
          fill="#161322"
          stroke="#B06AB3"
          strokeWidth={2}
        />
        <OrchidFlower cx={HUB_X} cy={HUB_Y} r={HUB_R} />
      </svg>

      {/* Text content */}
      <div className="flex-1 min-w-0 text-center lg:text-left">
        <p className="text-xs font-semibold tracking-widest uppercase text-orchid-accent mb-3">
          Intelligence as a Service
        </p>
        <h2 className="text-2xl font-bold text-orchid-text mb-3 leading-snug">
          One agentic core.<br />
          Three ways to expose it.
        </h2>
        <p className="text-orchid-muted text-sm leading-relaxed mb-6 max-w-md mx-auto lg:mx-0">
          Orchestrator Index builds the intelligence layer on top of any LLM. Your agents, tools,
          and skills are defined once — then exposed as an MCP gateway, a REST API, or embedded
          directly in Python code.
        </p>
        <div className="flex flex-col sm:flex-row gap-3 justify-center lg:justify-start">
          {SPOKES.map((spoke) => (
            <div
              key={spoke.label}
              className="flex items-center gap-2 px-3 py-2 rounded-lg border border-orchid-border bg-orchid-surface/50 text-sm"
            >
              <span className="w-2 h-2 rounded-full bg-orchid-accent" />
              <span className="text-orchid-text font-medium">{spoke.label}</span>
              <span className="text-orchid-muted text-xs">— {spoke.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
