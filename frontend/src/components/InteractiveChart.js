import React, { useRef, useEffect } from 'react';

export default function InteractiveChart({ algoId, points = [], mesh = [], line = [], centroids = [], axes = [] }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI screens
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear Canvas
    ctx.fillStyle = '#080c1a';
    ctx.fillRect(0, 0, width, height);

    // If no data points, draw placeholder guide
    if (points.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('點擊「擬合與更新模型」按鈕以生成動態圖表', width / 2, height / 2);
      return;
    }

    // Determine coordinate bounds for scaling
    let minX = -3, maxX = 3, minY = -3, maxY = 3;

    if (algoId === 'linear_regression') {
      // 1D regression mapping (x vs y)
      const allX = points.map(p => p.x1);
      const allY = points.map(p => p.label); // target y is stored in label
      
      minX = Math.min(...allX) - 1;
      maxX = Math.max(...allX) + 1;
      minY = Math.min(...allY) - 5;
      maxY = Math.max(...allY) + 5;
    } else {
      // 2D space mapping (x1 vs x2)
      const allX1 = points.map(p => p.x1);
      const allX2 = points.map(p => p.x2);
      
      minX = Math.min(...allX1) - 0.5;
      maxX = Math.max(...allX1) + 0.5;
      minY = Math.min(...allX2) - 0.5;
      maxY = Math.max(...allX2) + 0.5;
    }

    // Helper functions to scale values to canvas pixels
    const scaleX = (val) => {
      return ((val - minX) / (maxX - minX)) * (width - 60) + 30;
    };

    const scaleY = (val) => {
      // Flip Y axis for screen space
      return height - (((val - minY) / (maxY - minY)) * (height - 60) + 30);
    };

    // Draw Grid Lines (ticks)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 6; i++) {
      // vertical grid line
      const vx = (width / 6) * i;
      ctx.beginPath();
      ctx.moveTo(vx, 0);
      ctx.lineTo(vx, height);
      ctx.stroke();

      // horizontal grid line
      const hy = (height / 6) * i;
      ctx.beginPath();
      ctx.moveTo(0, hy);
      ctx.lineTo(width, hy);
      ctx.stroke();
    }

    // 1. Draw Decision Boundary Grid (Mesh)
    if (mesh && mesh.length > 0) {
      // We render a density grid. The mesh points are structured on a 40x40 grid.
      // Find cells and compute cell sizes
      const cellW = (width - 60) / 40 + 1;
      const cellH = (height - 60) / 40 + 1;

      mesh.forEach((m) => {
        const cx = scaleX(m.x1) - cellW / 2;
        const cy = scaleY(m.x2) - cellH / 2;

        // Determine background shading color based on model output value
        // If it is binary class, val is the probability or hard label (0 or 1)
        let fillColor = 'transparent';
        if (m.val !== undefined) {
          // If probability
          if (m.val >= 0 && m.val <= 1) {
            // Gradient fill from Coral (class 0) to Blue (class 1)
            const r = Math.round(244 * (1 - m.val) + 56 * m.val);
            const g = Math.round(63 * (1 - m.val) + 189 * m.val);
            const b = Math.round(94 * (1 - m.val) + 248 * m.val);
            fillColor = `rgba(${r}, ${g}, ${b}, 0.16)`;
          } else {
            // Hard labels (multiclass, 0, 1, 2)
            if (m.val === 0) fillColor = 'rgba(244, 63, 94, 0.12)';
            else if (m.val === 1) fillColor = 'rgba(56, 189, 248, 0.12)';
            else fillColor = 'rgba(168, 85, 247, 0.12)';
          }
        }

        ctx.fillStyle = fillColor;
        ctx.fillRect(cx, cy, cellW, cellH);
      });
    }

    // 2. Draw Regression Line (if linear regression)
    if (algoId === 'linear_regression' && line && line.length > 0) {
      ctx.beginPath();
      ctx.lineWidth = 3;
      const grad = ctx.createLinearGradient(0, 0, width, 0);
      grad.addColorStop(0, '#38bdf8');
      grad.addColorStop(1, '#a855f7');
      ctx.strokeStyle = grad;

      line.forEach((pt, idx) => {
        const px = scaleX(pt.x1);
        const py = scaleY(pt.y);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.shadowBlur = 10;
      ctx.shadowColor = 'rgba(56, 189, 248, 0.4)';
      ctx.stroke();
      
      // Reset shadow
      ctx.shadowBlur = 0;
    }

    // 3. Draw PCA Eigenvectors
    if (algoId === 'pca' && axes && axes.length > 0) {
      axes.forEach((axis, idx) => {
        const sx = scaleX(axis.start.x1);
        const sy = scaleY(axis.start.x2);
        const ex = scaleX(axis.end.x1);
        const ey = scaleY(axis.end.x2);

        // Vector arrow line
        ctx.beginPath();
        ctx.lineWidth = idx === 0 ? 3 : 2;
        ctx.strokeStyle = idx === 0 ? '#38bdf8' : '#a855f7';
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();

        // Draw arrow head
        const angle = Math.atan2(ey - sy, ex - sx);
        ctx.beginPath();
        ctx.fillStyle = idx === 0 ? '#38bdf8' : '#a855f7';
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - 10 * Math.cos(angle - Math.PI / 6), ey - 10 * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(ex - 10 * Math.cos(angle + Math.PI / 6), ey - 10 * Math.sin(angle + Math.PI / 6));
        ctx.fill();

        // Vector labels
        ctx.fillStyle = '#f8fafc';
        ctx.font = '11px Inter, sans-serif';
        const labelOffset = 15;
        ctx.fillText(`PC${idx + 1} (${Math.round(axis.variance_ratio * 100)}%)`, ex + Math.cos(angle) * labelOffset, ey + Math.sin(angle) * labelOffset);
      });
    }

    // 4. Draw Scatter Points
    points.forEach((pt) => {
      const px = scaleX(pt.x1);
      const py = algoId === 'linear_regression' ? scaleY(pt.label) : scaleY(pt.x2);

      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);

      // Determine dot color
      if (algoId === 'linear_regression') {
        ctx.fillStyle = '#38bdf8';
        ctx.strokeStyle = '#ffffff';
      } else {
        // Labeled colors
        if (pt.label === 0) {
          ctx.fillStyle = '#f43f5e'; // Coral Red
          ctx.strokeStyle = '#ffffff';
        } else if (pt.label === 1) {
          ctx.fillStyle = '#38bdf8'; // Sky Blue
          ctx.strokeStyle = '#ffffff';
        } else if (pt.label === 2) {
          ctx.fillStyle = '#a855f7'; // Purple
          ctx.strokeStyle = '#ffffff';
        } else {
          ctx.fillStyle = '#64748b'; // Gray fallback
          ctx.strokeStyle = '#ffffff';
        }
      }

      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();
    });

    // 5. Draw Centroids (for K-Means)
    if (algoId === 'kmeans' && centroids && centroids.length > 0) {
      centroids.forEach((c) => {
        const cx = scaleX(c.x1);
        const cy = scaleY(c.x2);

        // Outer glow circle
        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.fill();

        // Inner marker star/cross
        ctx.beginPath();
        ctx.arc(cx, cy, 7, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = c.label === 0 ? '#f43f5e' : c.label === 1 ? '#38bdf8' : '#a855f7';
        ctx.fill();
        ctx.stroke();

        // Centroid symbol
        ctx.fillStyle = '#060913';
        ctx.font = 'bold 9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('C', cx, cy);
      });
    }

  }, [algoId, points, mesh, line, centroids, axes]);

  return (
    <div className="canvas-wrapper">
      <canvas ref={canvasRef} className="interactive-canvas" />
    </div>
  );
}
