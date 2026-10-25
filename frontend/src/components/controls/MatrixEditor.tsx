/** Convolution-kernel editor with a live frequency response.
 *
 *  The response plot is the point: it shows whether the kernel you typed is a
 *  low-pass, a high-pass or something with awkward sidelobes, before you look
 *  at the image at all.
 */
import { Grid3x3, RotateCcw } from 'lucide-react'
import * as React from 'react'

import {
  Button, Input, Label, Popover, PopoverContent, PopoverTrigger, Switch,
  Tooltip,
} from '@/components/ui'
import type { ParamSpec } from '@/lib/types'
import { cn, fmtNum } from '@/lib/utils'

export interface MatrixEditorProps {
  spec: ParamSpec
  value: unknown
  onChange: (v: unknown) => void
  onBeginEdit?: () => void
}

/** |DFT| of the kernel on a zero-padded grid, as a coarse magnitude map. */
function frequencyResponse(k: number[][], n = 32): number[][] {
  const out: number[][] = []
  let maxV = 0
  for (let v = 0; v < n; v++) {
    const row: number[] = []
    for (let u = 0; u < n; u++) {
      // Centre the output so DC lands in the middle, as fftshift would.
      const fu = (u - n / 2) / n
      const fv = (v - n / 2) / n
      let re = 0
      let im = 0
      const cy = (k.length - 1) / 2
      const cx = (k[0].length - 1) / 2
      for (let y = 0; y < k.length; y++) {
        for (let x = 0; x < k[0].length; x++) {
          const phase = -2 * Math.PI * (fu * (x - cx) + fv * (y - cy))
          re += k[y][x] * Math.cos(phase)
          im += k[y][x] * Math.sin(phase)
        }
      }
      const mag = Math.hypot(re, im)
      maxV = Math.max(maxV, mag)
      row.push(mag)
    }
    out.push(row)
  }
  return out.map((r) => r.map((v) => (maxV > 0 ? v / maxV : 0)))
}

export function MatrixEditor({ spec, value, onChange, onBeginEdit }: MatrixEditorProps) {
  const matrix: number[][] = React.useMemo(() => {
    const raw = value as number[][] | undefined
    if (Array.isArray(raw) && Array.isArray(raw[0])) return raw.map((r) => [...r])
    const d = spec.default as number[][] | undefined
    if (Array.isArray(d) && Array.isArray(d[0])) return d.map((r) => [...r])
    return [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
  }, [value, spec.default])

  const [symmetry, setSymmetry] = React.useState(false)
  const rows = matrix.length
  const cols = matrix[0]?.length ?? 3
  const sum = matrix.flat().reduce((a, b) => a + b, 0)
  const response = React.useMemo(() => frequencyResponse(matrix), [matrix])
  const dcGain = response[16]?.[16] ?? 0
  const isHighPass = Math.abs(sum) < 0.5

  const setCell = (r: number, c: number, v: number) => {
    const next = matrix.map((row) => [...row])
    next[r][c] = v
    if (symmetry) {
      // Mirror into the three symmetric positions, keeping the kernel
      // isotropic -- which is usually what you want for a blur or a Laplacian.
      next[rows - 1 - r][c] = v
      next[r][cols - 1 - c] = v
      next[rows - 1 - r][cols - 1 - c] = v
    }
    onChange(next)
  }

  const resize = (n: number) => {
    const next: number[][] = Array.from({ length: n }, (_, r) =>
      Array.from({ length: n }, (_, c) => {
        const or = r - (n - rows) / 2
        const oc = c - (n - cols) / 2
        return Number.isInteger(or) && Number.isInteger(oc) && matrix[or]?.[oc] !== undefined
          ? matrix[or][oc]
          : 0
      }),
    )
    // Preserve a centre tap if the old kernel had one and the new grid is empty.
    if (next.flat().every((v) => v === 0)) next[(n - 1) / 2][(n - 1) / 2] = 1
    onChange(next)
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Tooltip content={spec.help} side="right">
          <Label className="cursor-help">{spec.label}</Label>
        </Tooltip>
        <div className="flex items-center gap-1">
          {spec.presets && spec.presets.length > 0 && (
            <Popover>
              <PopoverTrigger asChild>
                <Button variant="ghost" size="xs">
                  <Grid3x3 className="h-3 w-3" />
                  Presets
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-52 p-1">
                <div className="max-h-72 overflow-y-auto scrollbar-thin">
                  {spec.presets.map((p) => (
                    <button
                      key={p.label}
                      onClick={() => {
                        onBeginEdit?.()
                        onChange(p.value)
                      }}
                      className="block w-full rounded px-2 py-1.5 text-left text-xs hover:bg-accent"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </PopoverContent>
            </Popover>
          )}
          <Tooltip content="Reset to default">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => onChange(spec.default)}
              aria-label="Reset kernel"
            >
              <RotateCcw className="h-3 w-3" />
            </Button>
          </Tooltip>
        </div>
      </div>

      <div className="flex gap-2">
        <div
          className="grid flex-1 gap-0.5"
          style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
        >
          {matrix.map((row, r) =>
            row.map((v, c) => {
              const isCentre = r === (rows - 1) / 2 && c === (cols - 1) / 2
              return (
                <Input
                  key={`${r}-${c}`}
                  type="number"
                  step={0.1}
                  value={v}
                  onFocus={onBeginEdit}
                  onChange={(e) => setCell(r, c, Number(e.target.value))}
                  className={cn(
                    'h-6 px-0.5 text-center text-[10px] tabular',
                    isCentre && 'border-primary/60 font-semibold',
                  )}
                  aria-label={`Kernel ${r},${c}`}
                />
              )
            }),
          )}
        </div>

        {/* Frequency response: bright centre = passes low frequencies. */}
        <Tooltip
          content={
            isHighPass
              ? 'High-pass: the centre (DC) is dark, so flat areas are removed and edges remain.'
              : 'Low-pass: the centre (DC) is bright, so the average level is preserved.'
          }
        >
          <div className="shrink-0 cursor-help">
            <svg
              viewBox="0 0 32 32"
              className="h-[72px] w-[72px] rounded border border-input"
              aria-label="Kernel frequency response"
            >
              {response.map((row, y) =>
                row.map((v, x) => (
                  <rect
                    key={`${x}-${y}`}
                    x={x}
                    y={y}
                    width={1}
                    height={1}
                    fill={`hsl(199 89% ${Math.round(8 + v * 60)}%)`}
                  />
                )),
              )}
            </svg>
            <div className="mt-0.5 text-center text-[9px] text-muted-foreground">response</div>
          </div>
        </Tooltip>
      </div>

      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-muted-foreground">Size</span>
          {[3, 5, 7, 9].map((n) => (
            <button
              key={n}
              onClick={() => {
                onBeginEdit?.()
                resize(n)
              }}
              className={cn(
                'rounded px-1.5 py-0.5 text-[10px] transition-colors',
                rows === n ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-accent',
              )}
            >
              {n}×{n}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <Tooltip content="Mirror edits into all four symmetric positions">
            <span className="cursor-help text-[10px] text-muted-foreground">sym</span>
          </Tooltip>
          <Switch checked={symmetry} onCheckedChange={setSymmetry} aria-label="Symmetry lock" />
        </div>
      </div>

      <div className="flex items-center justify-between rounded bg-muted/50 px-2 py-1 text-[10px] tabular text-muted-foreground">
        <span>sum {fmtNum(sum, 3)}</span>
        <span>DC gain {fmtNum(dcGain, 3)}</span>
        <span className={isHighPass ? 'text-sev-info' : 'text-sev-none'}>
          {isHighPass ? 'high-pass' : 'low-pass'}
        </span>
      </div>
    </div>
  )
}
