/** Live histogram. Drawn to a canvas rather than SVG: it redraws on every
 *  preview, and 256 bins x 4 channels as DOM nodes would thrash layout. */
import * as React from 'react'

import { Badge, Label, Tooltip } from '@/components/ui'
import type { PreviewStats } from '@/lib/types'
import { cn, fmtPct } from '@/lib/utils'

type Mode = 'luma' | 'rgb' | 'r' | 'g' | 'b'

const CHANNEL_COLORS: Record<string, string> = {
  luma: 'rgba(210, 220, 235, 0.85)',
  r: 'rgba(239, 68, 68, 0.75)',
  g: 'rgba(34, 197, 94, 0.75)',
  b: 'rgba(59, 130, 246, 0.75)',
}

export function Histogram({
  stats, height = 104, showStats = true, className,
}: {
  stats?: PreviewStats | null
  height?: number
  showStats?: boolean
  className?: string
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const [mode, setMode] = React.useState<Mode>('rgb')
  const [logY, setLogY] = React.useState(false)

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !stats) return
    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = height
    canvas.width = Math.max(1, Math.floor(w * dpr))
    canvas.height = Math.max(1, Math.floor(h * dpr))
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, w, h)

    const series: [string, number[]][] = []
    if (mode === 'rgb') {
      if (stats.histogram_r) series.push(['r', stats.histogram_r])
      if (stats.histogram_g) series.push(['g', stats.histogram_g])
      if (stats.histogram_b) series.push(['b', stats.histogram_b])
      if (!series.length) series.push(['luma', stats.histogram])
    } else if (mode === 'luma') {
      series.push(['luma', stats.histogram])
    } else {
      const key = `histogram_${mode}` as keyof PreviewStats
      const arr = stats[key] as number[] | undefined
      series.push([mode, arr ?? stats.histogram])
    }

    // Scale by a high percentile rather than the max: a single huge spike
    // (a large flat region) would otherwise flatten the whole plot.
    const all = series.flatMap(([, d]) => d)
    const sorted = [...all].sort((a, b) => a - b)
    const p995 = sorted[Math.floor(sorted.length * 0.995)] || 1
    const maxV = Math.max(p995, 1)
    const yOf = (v: number) => {
      const n = logY ? Math.log1p(v) / Math.log1p(maxV) : v / maxV
      return h - Math.min(n, 1) * (h - 2) - 1
    }

    // Zone guides at quarter points.
    ctx.strokeStyle = 'rgba(128,128,128,0.18)'
    ctx.lineWidth = 1
    for (const g of [0.25, 0.5, 0.75]) {
      ctx.beginPath()
      ctx.moveTo(g * w, 0)
      ctx.lineTo(g * w, h)
      ctx.stroke()
    }

    ctx.globalCompositeOperation = series.length > 1 ? 'lighter' : 'source-over'
    for (const [name, data] of series) {
      ctx.beginPath()
      ctx.moveTo(0, h)
      for (let i = 0; i < data.length; i++) {
        ctx.lineTo((i / (data.length - 1)) * w, yOf(data[i]))
      }
      ctx.lineTo(w, h)
      ctx.closePath()
      ctx.fillStyle = CHANNEL_COLORS[name] ?? CHANNEL_COLORS.luma
      ctx.fill()
    }
    ctx.globalCompositeOperation = 'source-over'

    // Clipping markers: the practical reason to look at a histogram at all.
    if (stats.clip_low > 0.001) {
      ctx.fillStyle = 'rgba(59,130,246,0.85)'
      ctx.fillRect(0, 0, 3, h)
    }
    if (stats.clip_high > 0.001) {
      ctx.fillStyle = 'rgba(239,68,68,0.85)'
      ctx.fillRect(w - 3, 0, 3, h)
    }
  }, [stats, mode, logY, height])

  return (
    <div className={cn('space-y-1.5', className)}>
      <div className="flex items-center justify-between gap-2">
        <Label>Histogram</Label>
        <div className="flex items-center gap-0.5">
          {(['rgb', 'luma', 'r', 'g', 'b'] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={cn(
                'rounded px-1.5 py-0.5 text-[10px] uppercase transition-colors',
                mode === m ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-accent',
              )}
            >
              {m}
            </button>
          ))}
          <Tooltip content="Logarithmic vertical axis, so small counts stay visible">
            <button
              onClick={() => setLogY((v) => !v)}
              className={cn(
                'ml-0.5 rounded px-1.5 py-0.5 text-[10px] transition-colors',
                logY ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-accent',
              )}
            >
              log
            </button>
          </Tooltip>
        </div>
      </div>

      <canvas
        ref={canvasRef}
        style={{ height, width: '100%' }}
        className="rounded border border-border/60 bg-background"
        aria-label="Intensity histogram"
      />

      {showStats && stats && (
        <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] tabular text-muted-foreground">
          <span>mean {stats.mean.toFixed(3)}</span>
          <span>std {stats.std.toFixed(3)}</span>
          <span>min {stats.min.toFixed(3)}</span>
          <span>max {stats.max.toFixed(3)}</span>
          <Tooltip content="Pixels crushed to black; detail there is unrecoverable.">
            <span className={cn('cursor-help', stats.clip_low > 0.01 && 'text-sev-info')}>
              shadow clip {fmtPct(stats.clip_low)}
            </span>
          </Tooltip>
          <Tooltip content="Pixels blown to white; detail there is unrecoverable.">
            <span className={cn('cursor-help', stats.clip_high > 0.005 && 'text-sev-severe')}>
              highlight clip {fmtPct(stats.clip_high)}
            </span>
          </Tooltip>
        </div>
      )}

      {!stats && (
        <div className="flex h-5 items-center justify-center">
          <Badge variant="outline">no data yet</Badge>
        </div>
      )}
    </div>
  )
}
