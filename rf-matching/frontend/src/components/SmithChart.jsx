import React, { useRef, useEffect } from 'react';

/**
 * Light-themed Canvas-based Smith Chart.
 * Shows per-port reflection coefficient from joint-optimize results.
 */
const COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'];

export default function SmithChart({ jointResults, loadedSNP, selectedPort }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !jointResults) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;
    const r = Math.min(W, H) * 0.42;

    ctx.clearRect(0, 0, W, H);
    // Light background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, W, H);

    // Draw Smith chart grid (constant resistance/reactance circles)
    drawSmithGrid(ctx, cx, cy, r);

    // Draw unit circle
    ctx.strokeStyle = 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    ctx.stroke();

    // Draw horizontal axis
    ctx.strokeStyle = 'rgba(0,0,0,0.12)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(cx - r, cy);
    ctx.lineTo(cx + r, cy);
    ctx.stroke();

    // Draw center point (perfect match)
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.beginPath();
    ctx.arc(cx, cy, 2, 0, 2 * Math.PI);
    ctx.fill();

    // Plot each port from joint results
    const { results_per_port = {} } = jointResults;
    Object.entries(results_per_port).forEach(([piStr, pr]) => {
      const pi = parseInt(piStr);
      const color = COLORS[pi % COLORS.length];
      const isSelected = selectedPort == null || selectedPort === pi;

      // Try to reconstruct gamma from s11_magnitude
      // We only have magnitude, not phase, so show as a point on the real axis
      // For a more accurate display, we'd need complex S11 data
      const mag = pr.s11_magnitude != null ? pr.s11_magnitude : 0;
      const s11db = pr.s11_db;

      // Plot on positive real axis (worst-case assumption if no phase info)
      // If we had s11_real/s11_imag we could plot exact position
      const px = cx - (1 - mag) * r * 0.5; // Approximate position
      const py = cy;

      if (!isSelected) return;

      // Draw point
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`P${pi + 1}`, px, py + 14);
      if (s11db != null) {
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.font = '9px sans-serif';
        ctx.fillText(`${s11db.toFixed(1)}dB`, px, py + 26);
      }
    });

    // Legend
    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('● Port center = S11 magnitude', 8, H - 6);

  }, [jointResults, loadedSNP, selectedPort]);

  if (!jointResults) {
    return (
      <div style={{
        height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--text-secondary)', fontSize: 12,
      }}>
        Select a port to view Smith Chart
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      width={300}
      height={280}
      style={{
        width: '100%',
        height: '100%',
        maxWidth: 340,
        border: '1px solid var(--border)',
        borderRadius: 6,
        background: '#fff',
      }}
    />
  );
}

function drawSmithGrid(ctx, cx, cy, r) {
  // Constant resistance circles
  const rVals = [0, 0.2, 0.5, 1, 2, 5, 10];
  for (const rv of rVals) {
    const rcx = cx + r * rv / (1 + rv);
    const rcR = r / (1 + rv);
    if (rcR < r * 3) {
      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.arc(rcx, cy, rcR, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }

  // Constant reactance arcs
  const xVals = [0.2, 0.5, 1, 2, 5];
  for (const xv of xVals) {
    for (const sign of [1, -1]) {
      const x = xv * sign;
      const arcR = r / Math.abs(x);
      const arcCy = cy + sign * r / x;

      if (arcR > 0 && arcR < r * 5) {
        ctx.strokeStyle = 'rgba(0,0,0,0.08)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        const startY = cy - r;
        const endY = cy + r;
        const clampedStart = Math.max(-r, Math.min(r, arcCy - arcR));
        const clampedEnd = Math.max(-r, Math.min(r, arcCy + arcR));
        const startAngle = Math.asin((clampedStart - arcCy) / arcR);
        const endAngle = Math.asin((clampedEnd - arcCy) / arcR);
        ctx.arc(cx, arcCy, arcR, startAngle, endAngle);
        ctx.stroke();
      }
    }
  }
}
