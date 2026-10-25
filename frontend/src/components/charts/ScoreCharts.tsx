/** Advisor charts: the health ring, the dimension radar and metric bars.
 *
 *  Colour encodes severity, but position and length carry the same
 *  information, so the charts stay readable in greyscale and for viewers with
 *  colour-vision deficiencies.
 */
import * as React from 'react'
import {
  PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart,
  ResponsiveContainer, Tooltip as RTooltip,
} from 'recharts'

import { Tooltip } from '@/components/ui'
import type { Scores, Verification } from '@/lib/types'
import { cn, fmtNum, scoreColor, titleCase } from '@/lib/utils'

export function HealthRing({
  score, size = 108, label = 'Image health',
}: {
  score: number
  size?: number
  label?: string
}) {
  const r = size / 2 - 9
  const circumference = 2 * Math.PI * r
  const dash = (score / 100) * circumference
  const color = scoreColor(score)
  return (
    <div className="flex flex-col items-center gap-1">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2} cy={size / 2} r={r}
            fill="none" stroke="hsl(var(--secondary))" strokeWidth={8}
          />
          <circle
            cx={size / 2} cy={size / 2} r={r}
            fill="none" stroke={color} strokeWidth={8} strokeLinecap="round"
            strokeDasharray={`${dash} ${circumference}`}
            style={{ transition: 'stroke-dasharray 400ms ease, stroke 400ms ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-semibold tabular leading-none" style={{ color }}>
            {Math.round(score)}
          </span>
          <span className="text-[9px] uppercase tracking-wide text-muted-foreground">/ 100</span>
        </div>
      </div>
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</span>
    </div>
  )
}

export function DimensionRadar({
  scores, verification, height = 230,
}: {
  scores: Scores
  verification?: Verification | null
  height?: number
}) {
  const data = React.useMemo(() => {
    const dims = Object.keys(scores.weights ?? scores.dimensions)
    return dims
      .filter((d) => scores.dimensions[d])
      .map((d) => {
        const after = verification?.dimensions.find((x) => x.dimension === d)
        return {
          dimension: titleCase(d).slice(0, 10),
          full: titleCase(d),
          before: scores.dimensions[d].score,
          after: after?.after,
        }
      })
  }, [scores, verification])

  const hasAfter = data.some((d) => typeof d.after === 'number')

  return (
    <ResponsiveContainer width="100%" height={height}>
      <RadarChart data={data} outerRadius="72%">
        <PolarGrid stroke="hsl(var(--border))" />
        <PolarAngleAxis
          dataKey="dimension"
          tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }}
        />
        <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
        <Radar
          name={hasAfter ? 'Before' : 'Score'}
          dataKey="before"
          stroke="hsl(var(--muted-foreground))"
          fill="hsl(var(--muted-foreground))"
          fillOpacity={hasAfter ? 0.12 : 0.28}
          strokeWidth={1.5}
          strokeDasharray={hasAfter ? '4 3' : undefined}
        />
        {hasAfter && (
          <Radar
            name="After"
            dataKey="after"
            stroke="hsl(var(--primary))"
            fill="hsl(var(--primary))"
            fillOpacity={0.26}
            strokeWidth={2}
          />
        )}
        <RTooltip
          contentStyle={{
            background: 'hsl(var(--popover))',
            border: '1px solid hsl(var(--border))',
            borderRadius: 6,
            fontSize: 11,
            padding: '6px 8px',
          }}
          // Recharts types the payload very loosely; read the full dimension
          // name off the row so the tooltip is not truncated like the axis.
          labelFormatter={(_label, payload) => {
            const row = (payload as { payload?: { full?: string } }[] | undefined)?.[0]
            return row?.payload?.full ?? ''
          }}
          formatter={(value) => (typeof value === 'number' ? value.toFixed(1) : String(value))}
        />
      </RadarChart>
    </ResponsiveContainer>
  )
}

export function DimensionBars({
  scores, verification, onSelect, selected,
}: {
  scores: Scores
  verification?: Verification | null
  onSelect?: (dim: string) => void
  selected?: string | null
}) {
  const dims = Object.keys(scores.weights ?? scores.dimensions).filter((d) => scores.dimensions[d])
  return (
    <div className="space-y-1">
      {dims.map((d) => {
        const s = scores.dimensions[d]
        const after = verification?.dimensions.find((x) => x.dimension === d)
        const delta = after ? after.delta : undefined
        return (
          <button
            key={d}
            onClick={() => onSelect?.(d)}
            className={cn(
              'flex w-full items-center gap-2 rounded px-1.5 py-1 text-left transition-colors',
              selected === d ? 'bg-accent' : 'hover:bg-accent/60',
            )}
          >
            <span className="w-[72px] shrink-0 truncate text-[11px]">{titleCase(d)}</span>
            <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-secondary">
              <div
                className="absolute inset-y-0 left-0 rounded-full transition-all"
                style={{ width: `${s.score}%`, background: scoreColor(s.score) }}
              />
              {after && (
                <div
                  className="absolute inset-y-0 w-0.5 bg-foreground/70"
                  style={{ left: `${after.after}%` }}
                />
              )}
            </div>
            <span className="w-7 shrink-0 text-right text-[11px] tabular" style={{ color: scoreColor(s.score) }}>
              {Math.round(s.score)}
            </span>
            {typeof delta === 'number' && Math.abs(delta) >= 0.5 && (
              <span
                className={cn(
                  'w-9 shrink-0 text-right text-[10px] tabular',
                  delta > 0 ? 'text-sev-none' : after?.expected_tradeoff ? 'text-muted-foreground' : 'text-sev-severe',
                )}
              >
                {delta > 0 ? '+' : ''}
                {delta.toFixed(1)}
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}

/** One metric with its value positioned against the expected band. */
export function MetricGauge({
  label, value, expected, help,
}: {
  label: string
  value: number
  expected: [number, number]
  help?: string
}) {
  const [lo, hi] = expected
  // Show a window around the expected band so the value's position is readable
  // even when it sits well outside it.
  const span = Math.max(hi - lo, 1e-6)
  const vmin = Math.min(lo - span * 0.6, value - span * 0.15)
  const vmax = Math.max(hi + span * 0.6, value + span * 0.15)
  const pos = ((value - vmin) / Math.max(vmax - vmin, 1e-9)) * 100
  const bandL = ((lo - vmin) / Math.max(vmax - vmin, 1e-9)) * 100
  const bandW = ((hi - lo) / Math.max(vmax - vmin, 1e-9)) * 100
  const inside = value >= lo && value <= hi

  return (
    <Tooltip content={help ?? `Expected ${fmtNum(lo)} to ${fmtNum(hi)}`}>
      <div className="cursor-help space-y-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className="truncate text-[10px] text-muted-foreground">{label}</span>
          <span
            className={cn('shrink-0 text-[11px] tabular', inside ? 'text-sev-none' : 'text-sev-moderate')}
          >
            {fmtNum(value)}
          </span>
        </div>
        <div className="relative h-1.5 overflow-hidden rounded-full bg-secondary">
          <div
            className="absolute inset-y-0 bg-sev-none/35"
            style={{ left: `${bandL}%`, width: `${bandW}%` }}
          />
          <div
            className={cn(
              'absolute inset-y-0 w-[3px] rounded',
              inside ? 'bg-sev-none' : 'bg-sev-moderate',
            )}
            style={{ left: `calc(${Math.min(Math.max(pos, 0), 100)}% - 1.5px)` }}
          />
        </div>
      </div>
    </Tooltip>
  )
}
