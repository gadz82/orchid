'use client';

import { useEffect, useRef } from 'react';

// Fibonacci spiral constants
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5)); // ≈ 2.399 rad
const N = 144; // 144 = Fibonacci number — natural petal count

// Five orchid shades cycling across the spiral
const SHADES: [number, number, number][] = [
  [176, 106, 179], // orchid-accent
  [212, 144, 215], // orchid-accent-glow
  [155, 89, 160],  // deeper
  [192, 112, 195], // mid
  [232, 168, 234], // pale
];

const LABEL_TEXTS = ['0.847', '∇ loss', '[-0.23, …]', 'cos θ', '768d', 'ε', '⟨v, w⟩', 'σ(Wx)'];

interface FloatingLabel {
  angle: number;
  rOffset: number;
  speed: number;
  text: string;
  opacity: number;
  targetOpacity: number;
  nextFlip: number;
}

export default function HeroCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Respect prefers-reduced-motion
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let rafId = 0;
    let tick = 0;
    let W = 0;
    let H = 0;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas!.getBoundingClientRect();
      W = rect.width;
      H = rect.height;
      canvas!.width = W * dpr;
      canvas!.height = H * dpr;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();

    // Floating label state
    const labels: FloatingLabel[] = LABEL_TEXTS.map((text, i) => ({
      angle: (i / LABEL_TEXTS.length) * Math.PI * 2,
      rOffset: Math.random() * 40,
      speed: (0.00025 + Math.random() * 0.00018) * (i % 2 === 0 ? 1 : -1),
      text,
      opacity: 0.1 + Math.random() * 0.25,
      targetOpacity: 0.1 + Math.random() * 0.35,
      nextFlip: Math.random() * 300,
    }));

    function drawFrame() {
      ctx!.clearRect(0, 0, W, H);

      const cx = W / 2;
      const cy = H / 2;
      // Spiral fills ~38% of the smaller dimension
      const maxR = Math.min(W, H) * 0.38;
      // Very slow global rotation
      const globalRot = prefersReduced ? 0 : tick * 0.00014;

      // ── Precompute particle positions ─────────────────────────
      const pts = new Float32Array(N * 3); // [x, y, nr] × N
      for (let i = 0; i < N; i++) {
        const angle = i * GOLDEN_ANGLE + globalRot;
        const nr = Math.sqrt(i / N); // 0..1
        // Gentle breathing pulse
        const pulse = prefersReduced ? 0 : Math.sin(tick * 0.0009 + i * 0.22) * 5;
        const r = nr * maxR + pulse;
        pts[i * 3 + 0] = cx + Math.cos(angle) * r;
        pts[i * 3 + 1] = cy + Math.sin(angle) * r;
        pts[i * 3 + 2] = nr;
      }

      // ── Connection lines ──────────────────────────────────────
      ctx!.lineWidth = 0.5;
      for (let i = 0; i < N; i++) {
        const xi = pts[i * 3];
        const yi = pts[i * 3 + 1];
        // Only check a neighbourhood window to stay O(N·k) not O(N²)
        for (let j = i + 1; j < Math.min(i + 14, N); j++) {
          const xj = pts[j * 3];
          const yj = pts[j * 3 + 1];
          const d = Math.hypot(xj - xi, yj - yi);
          if (d < 70) {
            const alpha = (1 - d / 70) * 0.26;
            ctx!.strokeStyle = `rgba(176,106,179,${alpha.toFixed(3)})`;
            ctx!.beginPath();
            ctx!.moveTo(xi, yi);
            ctx!.lineTo(xj, yj);
            ctx!.stroke();
          }
        }
      }

      // ── Particles ─────────────────────────────────────────────
      for (let i = 0; i < N; i++) {
        const x = pts[i * 3];
        const y = pts[i * 3 + 1];
        const nr = pts[i * 3 + 2];
        // Center particles are slightly larger
        const size = 1.1 + (1 - nr) * 2.4;
        const alpha = 0.35 + nr * 0.55;

        const [r, g, b] = SHADES[i % SHADES.length];

        // Outer glow halo
        ctx!.beginPath();
        ctx!.arc(x, y, size * 3.2, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(${r},${g},${b},${(alpha * 0.08).toFixed(3)})`;
        ctx!.fill();

        // Inner glow ring
        ctx!.beginPath();
        ctx!.arc(x, y, size * 1.9, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(${r},${g},${b},${(alpha * 0.22).toFixed(3)})`;
        ctx!.fill();

        // Core
        ctx!.beginPath();
        ctx!.arc(x, y, size, 0, Math.PI * 2);
        ctx!.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
        ctx!.fill();
      }

      // ── Floating labels ───────────────────────────────────────
      ctx!.font = '11px ui-monospace, "JetBrains Mono", monospace';
      ctx!.textAlign = 'center';
      ctx!.textBaseline = 'middle';

      const labelBaseR = maxR * 1.22;
      for (const lbl of labels) {
        if (!prefersReduced) {
          lbl.angle += lbl.speed;
          if (tick >= lbl.nextFlip) {
            lbl.targetOpacity = 0.07 + Math.random() * 0.32;
            lbl.nextFlip = tick + 220 + Math.random() * 380;
          }
          lbl.opacity += (lbl.targetOpacity - lbl.opacity) * 0.012;
        }
        const drift = prefersReduced ? 0 : Math.sin(tick * 0.0007 + lbl.rOffset) * 14;
        const lx = cx + Math.cos(lbl.angle) * (labelBaseR + drift);
        const ly = cy + Math.sin(lbl.angle) * (labelBaseR + drift * 0.6);
        ctx!.fillStyle = `rgba(212,144,215,${lbl.opacity.toFixed(3)})`;
        ctx!.fillText(lbl.text, lx, ly);
      }

      tick++;
      rafId = requestAnimationFrame(drawFrame);
    }

    drawFrame();

    return () => {
      cancelAnimationFrame(rafId);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      aria-hidden="true"
    />
  );
}
