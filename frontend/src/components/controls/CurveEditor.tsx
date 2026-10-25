/** Tone-curve editor. Mirrors the backend's monotone cubic interpolation
 *  (ops/_util.monotone_spline) so the preview line matches what the server
 *  will actually apply. */
import { RotateCcw } from 'lucide-react'
import * as React from 'react'

import { Button, Label, Tooltip } from '@/components/ui'
import type { ParamSpec } from '@/lib/types'
import { clamp } from '@/lib/utils'

type Pt = [number, number]

/** Fritsch-Carlson monotone cubic: cannot overshoot, so a curve can never
 *  invert locally and solarise the image. Same algorithm as the backend. */
function monotoneSpline(points: Pt[], n = 256): number[] {
  const pts = [...points].sort((a, b) => a[0] - b[0])
  const xs: number[] = []
  const ys: number[] = []
  for (const [x, y] of pts) {
    if (xs.length === 0 || x - xs[xs.length - 1] > 1e-9) {
      xs.push(x)
      ys.push(y)
    }
  }
  if (xs.length < 2) return Array.from({ length: n }, (_, i) => i / (n - 1))

  const k = xs.length
  const h: number[] = []
  const delta: number[] = []
  for (let i = 0; i < k - 1; i++) {
    h.push(xs[i + 1] - xs[i])
    delta.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]))
  }
  const m = new Array<number>(k).fill(0)
  m[0] = delta[0]
  m[k - 1] = delta[k - 2]
  for (let i = 1; i < k - 1; i++) m[i] = (delta[i - 1] + delta[i]) / 2
  for (let i = 0; i < k - 1; i++) {
    if (Math.abs(delta[i]) < 1e-12) {
      m[i] = 0
      m[i + 1] = 0
    } else {
      const a = m[i] / delta[i]
      const b = m[i + 1] / delta[i]
      const s = a * a + b * b
      if (s > 9) {
        const t = 3 / Math.sqrt(s)
        m[i] = t * a * delta[i]
        m[i + 1] = t * b * delta[i]
      }
    }
  }

  const out: number[] = []
  for (let i = 0; i < n; i++) {
    const x = i / (n - 1)
    let seg = 0
    while (seg < k - 2 && xs[seg + 1] < x) seg++
    const hh = h[seg]
    const t = clamp((x - xs[seg]) / hh, 0, 1)
    const t2 = t * t
    const t3 = t2 * t
    const h00 = 2 * t3 - 3 * t2 + 1
    const h10 = t3 - 2 * t2 + t
    const h01 = -2 * t3 + 3 * t2
    const h11 = t3 - t2
    out.push(
      clamp(h00 * ys[seg] + h10 * hh * m[seg] + h01 * ys[seg + 1] + h11 * hh * m[seg + 1], 0, 1),
    )
  }
  return out
}

export interface CurveEditorProps {
  spec: ParamSpec
  value: unknown
  onChange: (v: unknown) => void
  onBeginEdit?: () => void
  /** Histogram drawn behind the curve, so you can see what you are shaping. */
  histogram?: number[]
}

export function CurveEditor({ spec, value, onChange, onBeginEdit, histogram }: CurveEditorProps) {
  const size = 188
  const svgRef = React.useRef<SVGSVGElement>(null)
  const [dragIndex, setDragIndex] = React.useState<number | null>(null)

  const points: Pt[] = React.useMemo(() => {
    const raw = Array.isArray(value) ? (value as number[][]) : []
    const pts = raw
      .filter((p) => Array.isArray(p) && p.length >= 2)
      .map((p) => [clamp(p[0], 0, 1), clamp(p[1], 0, 1)] as Pt)
    return pts.length >= 2 ? pts.sort((a, b) => a[0] - b[0]) : [[0, 0], [1, 1]]
  }, [value])

  const lut = React.useMemo(() => monotoneSpline(points, 128), [points])

  const toSvg = (p: Pt): Pt => [p[0] * size, (1 - p[1]) * size]
  const fromEvent = (e: PointerEvent | React.PointerEvent): Pt => {
    const rect = svgRef.current!.getBoundingClientRect()
    return [
      clamp((e.clientX - rect.left) / rect.width, 0, 1),
      clamp(1 - (e.clientY - rect.top) / rect.height, 0, 1),
    ]
  }

  const commit = (pts: Pt[]) => onChange(pts.map((p): Pt => [p[0], p[1]]))

  const onPointerDownBg = (e: React.PointerEvent) => {
    const [x, y] = fromEvent(e)
    // Clicking the background adds a control point at that position.
    onBeginEdit?.()
    const next: Pt[] = [...points, [x, y] as Pt].sort((a, b) => a[0] - b[0])
    commit(next)
    setDragIndex(next.findIndex((p) => p[0] === x && p[1] === y))
  }

  React.useEffect(() => {
    if (dragIndex === null) return
    const move = (e: PointerEvent) => {
      const [x, y] = fromEvent(e)
      const next = points.map((p, i) => (i === dragIndex ? ([x, y] as Pt) : p))
      // Endpoints keep their x so the curve always spans the full range.
      if (dragIndex === 0) next[0] = [0, y]
      if (dragIndex === points.length - 1) next[points.length - 1] = [1, y]
      commit(next.sort((a, b) => a[0] - b[0]))
    }
    const up = () => setDragIndex(null)
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [dragIndex, points]) // eslint-disable-line react-hooks/exhaustive-deps

  const curvePath = lut
    .map((y, i) => {
      const [sx, sy] = toSvg([i / (lut.length - 1), y])
      return `${i === 0 ? 'M' : 'L'}${sx.toFixed(1)},${sy.toFixed(1)}`
    })
    .join(' ')

  const histMax = histogram?.length ? Math.max(...histogram, 1) : 1
  const histPath = histogram?.length
    ? histogram
        .map((v, i) => {
          const x = (i / (histogram.length - 1)) * size
          const y = size - (v / histMax) * size * 0.9
          return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
        })
        .join(' ') + ` L${size},${size} L0,${size} Z`
    : null

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <Tooltip content={spec.help} side="right">
          <Label className="cursor-help">{spec.label}</Label>
        </Tooltip>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-muted-foreground">{points.length} pts</span>
          <Tooltip content="Reset to a straight line">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => commit([[0, 0], [1, 1]])}
              aria-label="Reset curve"
            >
              <RotateCcw className="h-3 w-3" />
            </Button>
          </Tooltip>
        </div>
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${size} ${size}`}
        className="w-full cursor-crosshair touch-none rounded border border-input bg-background"
        onPointerDown={onPointerDownBg}
        role="application"
        aria-label={`${spec.label} curve editor`}
      >
        {histPath && <path d={histPath} fill="hsl(var(--muted-foreground))" opacity={0.2} />}
        {[0.25, 0.5, 0.75].map((g) => (
          <g key={g} stroke="hsl(var(--border))" strokeWidth={0.5} opacity={0.7}>
            <line x1={g * size} y1={0} x2={g * size} y2={size} />
            <line x1={0} y1={g * size} x2={size} y2={g * size} />
          </g>
        ))}
        <line
          x1={0} y1={size} x2={size} y2={0}
          stroke="hsl(var(--border))" strokeWidth={0.75} strokeDasharray="3 3"
        />
        <path d={curvePath} fill="none" stroke="hsl(var(--primary))" strokeWidth={1.75} />
        {points.map((p, i) => {
          const [sx, sy] = toSvg(p)
          return (
            <circle
              key={i}
              cx={sx}
              cy={sy}
              r={4.5}
              fill={dragIndex === i ? 'hsl(var(--primary))' : 'hsl(var(--background))'}
              stroke="hsl(var(--primary))"
              strokeWidth={1.75}
              className="cursor-grab"
              onPointerDown={(e) => {
                e.stopPropagation()
                onBeginEdit?.()
                // Alt-click removes an interior point; endpoints must remain.
                if (e.altKey && i > 0 && i < points.length - 1) {
                  commit(points.filter((_, j) => j !== i))
                  return
                }
                setDragIndex(i)
              }}
            />
          )
        })}
      </svg>
      <p className="text-[10px] leading-snug text-muted-foreground">
        Click to add a point, drag to shape, alt-click to remove. Monotone cubic
        interpolation, so the curve cannot overshoot.
      </p>
    </div>
  )
}
