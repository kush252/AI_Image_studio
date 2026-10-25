/** Schema-driven parameter controls. PLAN.md D1 and Section 5.
 *
 *  One renderer per ParamSpec.kind. Because every operator declares its
 *  parameters in the backend registry, adding an operator needs no change
 *  here -- which is the whole point of the registry design.
 */
import { Dices, RotateCcw, Sparkles, Upload } from 'lucide-react'
import * as React from 'react'

import {
  Button, Input, Label, Select, SelectContent, SelectItem, SelectTrigger,
  SelectValue, Slider, Switch, Tooltip,
} from '@/components/ui'
import type { ParamSpec } from '@/lib/types'
import { cn, fmtNum, snapToSpec, titleCase } from '@/lib/utils'

import { CurveEditor } from './CurveEditor'
import { MatrixEditor } from './MatrixEditor'

export interface ParamControlProps {
  spec: ParamSpec
  value: unknown
  onChange: (value: unknown) => void
  /** Called once when a drag begins, so history records one entry per gesture. */
  onBeginEdit?: () => void
  onAuto?: () => void
  autoBusy?: boolean
  imageOptions?: { id: string; label: string }[]
  disabled?: boolean
}

function ParamHeader({
  spec, children, onReset, onAuto, autoBusy,
}: {
  spec: ParamSpec
  children?: React.ReactNode
  onReset?: () => void
  onAuto?: () => void
  autoBusy?: boolean
}) {
  return (
    <div className="flex min-h-5 items-center justify-between gap-2">
      <Tooltip content={spec.help} side="right">
        <Label
          className={cn(
            'cursor-help truncate',
            spec.scale_aware && 'after:ml-0.5 after:text-primary/70 after:content-["*"]',
          )}
        >
          {spec.label}
          {spec.unit ? <span className="ml-1 opacity-60">({spec.unit})</span> : null}
        </Label>
      </Tooltip>
      <div className="flex shrink-0 items-center gap-1">
        {children}
        {onAuto && (
          <Tooltip content="Compute this from the image">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={onAuto}
              disabled={autoBusy}
              aria-label={`Auto ${spec.label}`}
            >
              <Sparkles className={cn('h-3 w-3', autoBusy && 'animate-pulse')} />
            </Button>
          </Tooltip>
        )}
        {onReset && (
          <Tooltip content="Reset to default">
            <Button variant="ghost" size="icon-sm" onClick={onReset} aria-label={`Reset ${spec.label}`}>
              <RotateCcw className="h-3 w-3" />
            </Button>
          </Tooltip>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ numeric */

function NumericParam({ spec, value, onChange, onBeginEdit, onAuto, autoBusy, disabled }: ParamControlProps) {
  const num = typeof value === 'number' ? value : Number(spec.default ?? 0)
  const [text, setText] = React.useState<string | null>(null)
  const min = spec.min ?? 0
  const max = spec.max ?? 1
  const step = spec.kind === 'float' ? (spec.step ?? 0.01) : 1

  // A log-scaled slider maps its position through a log so that, for example,
  // gamma 0.1-5 gives even resolution either side of 1 instead of cramming
  // everything below 1 into the first 18% of travel.
  const logScale = Boolean(spec.log_scale) && min > 0
  const toSlider = (v: number) =>
    logScale ? (Math.log(v / min) / Math.log(max / min)) * 1000 : v
  const fromSlider = (p: number) =>
    logScale ? min * Math.pow(max / min, p / 1000) : p

  const commitText = () => {
    if (text === null) return
    const parsed = Number(text)
    setText(null)
    if (Number.isFinite(parsed)) onChange(snapToSpec(spec, parsed))
  }

  return (
    <div className="space-y-1.5">
      <ParamHeader
        spec={spec}
        onReset={() => onChange(spec.default)}
        onAuto={spec.auto ? onAuto : undefined}
        autoBusy={autoBusy}
      >
        <Input
          className="h-5 w-16 px-1 text-right text-[11px] tabular"
          value={text ?? fmtNum(num, spec.kind === 'float' ? 3 : 0)}
          onChange={(e) => setText(e.target.value)}
          onBlur={commitText}
          onKeyDown={(e) => {
            if (e.key === 'Enter') { commitText(); (e.target as HTMLInputElement).blur() }
            if (e.key === 'Escape') setText(null)
          }}
          disabled={disabled}
          aria-label={`${spec.label} value`}
        />
      </ParamHeader>
      <Slider
        value={[toSlider(num)]}
        min={logScale ? 0 : min}
        max={logScale ? 1000 : max}
        step={logScale ? 1 : step}
        disabled={disabled}
        onPointerDown={onBeginEdit}
        onValueChange={([v]) => onChange(snapToSpec(spec, fromSlider(v)))}
        // Double-clicking the track returns the param to its default, which is
        // the convention in every professional editor.
        onDoubleClick={() => onChange(spec.default)}
        aria-label={spec.label}
      />
    </div>
  )
}

/* --------------------------------------------------------------------- misc */

function BoolParam({ spec, value, onChange, disabled }: ParamControlProps) {
  return (
    <div className="flex items-center justify-between gap-2 py-0.5">
      <Tooltip content={spec.help} side="right">
        <Label className="cursor-help">{spec.label}</Label>
      </Tooltip>
      <Switch
        checked={Boolean(value)}
        onCheckedChange={(v) => onChange(v)}
        disabled={disabled}
        aria-label={spec.label}
      />
    </div>
  )
}

function EnumParam({ spec, value, onChange, disabled }: ParamControlProps) {
  const current = String(value ?? spec.default ?? '')
  const opt = spec.options?.find((o) => o.value === current)
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec} onReset={() => onChange(spec.default)} />
      <Select value={current} onValueChange={onChange} disabled={disabled}>
        <SelectTrigger>
          <SelectValue placeholder="Select..." />
        </SelectTrigger>
        <SelectContent>
          {spec.options?.map((o) => (
            <SelectItem key={o.value} value={o.value}>
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {opt?.help && (
        <p className="pl-0.5 text-[10px] leading-snug text-muted-foreground">{opt.help}</p>
      )}
    </div>
  )
}

function RangeParam({ spec, value, onChange, onBeginEdit, disabled }: ParamControlProps) {
  const arr = Array.isArray(value) ? (value as number[]) : [0, 1]
  const min = spec.min ?? 0
  const max = spec.max ?? 1
  return (
    <div className="space-y-1.5">
      <ParamHeader spec={spec} onReset={() => onChange(spec.default)}>
        <span className="tabular text-[11px] text-muted-foreground">
          {fmtNum(arr[0], 2)} – {fmtNum(arr[1], 2)}
        </span>
      </ParamHeader>
      <Slider
        value={[arr[0], arr[1]]}
        min={min}
        max={max}
        step={spec.step ?? 0.01}
        minStepsBetweenThumbs={0}
        disabled={disabled}
        onPointerDown={onBeginEdit}
        onValueChange={(v) => onChange([v[0], v[1]])}
        aria-label={spec.label}
      />
    </div>
  )
}

function ColorParam({ spec, value, onChange, disabled }: ParamControlProps) {
  const arr = Array.isArray(value) ? (value as number[]) : [1, 1, 1]
  const hex =
    '#' +
    arr
      .slice(0, 3)
      .map((c) => Math.round(Math.min(Math.max(c, 0), 1) * 255).toString(16).padStart(2, '0'))
      .join('')
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec} onReset={() => onChange(spec.default)} />
      <div className="flex items-center gap-2">
        <input
          type="color"
          value={hex}
          disabled={disabled}
          onChange={(e) => {
            const h = e.target.value.slice(1)
            onChange([0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255))
          }}
          className="h-7 w-10 cursor-pointer rounded border border-input bg-background p-0.5"
          aria-label={spec.label}
        />
        <span className="tabular text-[11px] text-muted-foreground">{hex.toUpperCase()}</span>
      </div>
    </div>
  )
}

function PointParam({ spec, value, onChange, disabled }: ParamControlProps) {
  const p = Array.isArray(value) ? (value as number[]) : [0.5, 0.5]
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec} onReset={() => onChange(spec.default)}>
        <span className="text-[10px] text-muted-foreground">drag on canvas</span>
      </ParamHeader>
      <div className="grid grid-cols-2 gap-1.5">
        {(['x', 'y'] as const).map((axis, i) => (
          <div key={axis} className="flex items-center gap-1">
            <span className="w-3 text-[10px] text-muted-foreground">{axis}</span>
            <Input
              className="h-6 text-[11px] tabular"
              type="number"
              step={0.01}
              min={0}
              max={1}
              value={Number(p[i] ?? 0.5).toFixed(3)}
              disabled={disabled}
              onChange={(e) => {
                const next = [...p]
                next[i] = Number(e.target.value)
                onChange(next)
              }}
              aria-label={`${spec.label} ${axis}`}
            />
          </div>
        ))}
      </div>
    </div>
  )
}

function RectParam({ spec, value, onChange, disabled }: ParamControlProps) {
  const r = Array.isArray(value) ? (value as number[]) : [0, 0, 1, 1]
  const labels = ['x', 'y', 'w', 'h']
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec} onReset={() => onChange(spec.default)}>
        <span className="text-[10px] text-muted-foreground">drag on canvas</span>
      </ParamHeader>
      <div className="grid grid-cols-4 gap-1">
        {labels.map((l, i) => (
          <div key={l} className="space-y-0.5">
            <span className="block text-center text-[9px] text-muted-foreground">{l}</span>
            <Input
              className="h-6 px-1 text-center text-[10px] tabular"
              type="number"
              step={0.01}
              value={Number(r[i] ?? 0).toFixed(2)}
              disabled={disabled}
              onChange={(e) => {
                const next = [...r]
                next[i] = Number(e.target.value)
                onChange(next)
              }}
              aria-label={`${spec.label} ${l}`}
            />
          </div>
        ))}
      </div>
    </div>
  )
}

function PointsParam({ spec, value, onChange }: ParamControlProps) {
  const pts = Array.isArray(value) ? (value as number[][]) : []
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec} onReset={() => onChange(spec.default)}>
        <span className="text-[10px] text-muted-foreground">
          {pts.length} point{pts.length === 1 ? '' : 's'}
        </span>
      </ParamHeader>
      <p className="text-[10px] leading-snug text-muted-foreground">
        {spec.min_count && spec.min_count === spec.max_count
          ? `Drag the ${spec.min_count} handles on the canvas.`
          : 'Click the canvas to add points; click a handle to remove it.'}
      </p>
      {pts.length > 0 && (
        <Button variant="ghost" size="xs" onClick={() => onChange([])}>
          Clear points
        </Button>
      )}
    </div>
  )
}

function LineParam({ spec, value, onChange }: ParamControlProps) {
  const l = (Array.isArray(value) ? value : [[0.1, 0.5], [0.9, 0.5]]) as number[][]
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec} onReset={() => onChange(spec.default)}>
        <span className="text-[10px] text-muted-foreground">drag endpoints</span>
      </ParamHeader>
      <div className="tabular text-[10px] text-muted-foreground">
        ({l[0][0].toFixed(2)}, {l[0][1].toFixed(2)}) → ({l[1][0].toFixed(2)}, {l[1][1].toFixed(2)})
      </div>
    </div>
  )
}

function ImageRefParam({ spec, value, onChange, imageOptions, disabled }: ParamControlProps) {
  const opts = imageOptions ?? []
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec} onReset={() => onChange(null)} />
      {opts.length === 0 ? (
        <p className="rounded border border-dashed border-border px-2 py-1.5 text-[10px] leading-snug text-muted-foreground">
          Load a second image to use this operator.
        </p>
      ) : (
        <Select value={String(value ?? '')} onValueChange={onChange} disabled={disabled}>
          <SelectTrigger>
            <SelectValue placeholder="Select an image..." />
          </SelectTrigger>
          <SelectContent>
            {opts.map((o) => (
              <SelectItem key={o.id} value={o.id}>
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </div>
  )
}

function SeedParam({ spec, value, onChange }: ParamControlProps) {
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec}>
        <Tooltip content="New random seed">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => onChange(Math.floor(Math.random() * 2 ** 31))}
            aria-label="Randomise seed"
          >
            <Dices className="h-3 w-3" />
          </Button>
        </Tooltip>
      </ParamHeader>
      <Input
        className="tabular"
        type="number"
        value={Number(value ?? 0)}
        onChange={(e) => onChange(Math.max(0, Math.floor(Number(e.target.value))))}
        aria-label={spec.label}
      />
    </div>
  )
}

function LutFileParam({ spec, value, onChange }: ParamControlProps) {
  const inputRef = React.useRef<HTMLInputElement>(null)
  const loaded = Boolean(value)
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec} onReset={() => onChange(null)} />
      <input
        ref={inputRef}
        type="file"
        accept=".cube,text/plain"
        className="hidden"
        onChange={async (e) => {
          const f = e.target.files?.[0]
          if (!f) return
          const text = await f.text()
          onChange({ name: f.name, text })
        }}
      />
      <Button variant="outline" size="sm" className="w-full" onClick={() => inputRef.current?.click()}>
        <Upload className="h-3.5 w-3.5" />
        {loaded ? ((value as { name?: string }).name ?? 'LUT loaded') : 'Load .cube LUT'}
      </Button>
    </div>
  )
}

function StringParam({ spec, value, onChange, disabled }: ParamControlProps) {
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec} onReset={() => onChange(spec.default)} />
      <Input
        value={String(value ?? '')}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        aria-label={spec.label}
      />
    </div>
  )
}

function MaskRefParam({ spec, value, onChange }: ParamControlProps) {
  const has = Boolean(value)
  return (
    <div className="space-y-1">
      <ParamHeader spec={spec} onReset={() => onChange(null)} />
      <p className="rounded border border-dashed border-border px-2 py-1.5 text-[10px] leading-snug text-muted-foreground">
        {has
          ? 'A mask is attached. Clear it with the reset button.'
          : 'Build a mask with a Compose operator, then promote it here.'}
      </p>
    </div>
  )
}

function UnsupportedParam({ spec }: ParamControlProps) {
  return (
    <div className="space-y-1">
      <Label>{spec.label}</Label>
      <p className="rounded border border-dashed border-border px-2 py-1.5 text-[10px] leading-snug text-muted-foreground">
        This parameter is edited on the canvas ({titleCase(spec.kind)}).
      </p>
    </div>
  )
}

/* ---------------------------------------------------------------- dispatch */

export function ParamControl(props: ParamControlProps) {
  switch (props.spec.kind) {
    case 'float':
    case 'int':
    case 'odd_int':
      return <NumericParam {...props} />
    case 'bool':
      return <BoolParam {...props} />
    case 'enum':
      return <EnumParam {...props} />
    case 'range':
      return <RangeParam {...props} />
    case 'color':
      return <ColorParam {...props} />
    case 'point':
      return <PointParam {...props} />
    case 'points':
      return <PointsParam {...props} />
    case 'rect':
      return <RectParam {...props} />
    case 'line':
      return <LineParam {...props} />
    case 'matrix':
      return <MatrixEditor {...props} />
    case 'curve':
      return <CurveEditor {...props} />
    case 'image_ref':
      return <ImageRefParam {...props} />
    case 'mask_ref':
      return <MaskRefParam {...props} />
    case 'seed':
      return <SeedParam {...props} />
    case 'lut_file':
      return <LutFileParam {...props} />
    case 'string':
      return <StringParam {...props} />
    default:
      return <UnsupportedParam {...props} />
  }
}
