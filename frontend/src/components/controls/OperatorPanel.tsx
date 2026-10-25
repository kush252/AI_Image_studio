/** Renders every control for one operator from its schema, plus the Explain
 *  drawer (formula, theory, references) that Learn mode surfaces. */
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { BookOpen, ChevronDown, Info } from 'lucide-react'
import * as React from 'react'

import {
  Badge, Button, Label, Select, SelectContent, SelectItem, SelectTrigger,
  SelectValue, Separator, Slider, Tooltip,
} from '@/components/ui'
import { api } from '@/lib/api'
import type { OperationSpec, StackOp } from '@/lib/types'
import { cn, fmtNum, paramVisible, titleCase, wireOpsSafe } from '@/lib/utils'
import { useCatalogue, usePipeline, useSession, useUI } from '@/store'

import { CurveEditor } from './CurveEditor'
import { ParamControl } from './ParamControl'

export function Formula({ latex, display = true }: { latex: string; display?: boolean }) {
  const html = React.useMemo(() => {
    if (!latex) return ''
    try {
      return katex.renderToString(latex, {
        displayMode: display,
        throwOnError: false,
        // trust: false is the default; formulas come from our own operator
        // definitions, but there is no reason to widen it.
        strict: false,
        output: 'html',
      })
    } catch {
      return ''
    }
  }, [latex, display])
  if (!html) return null
  return (
    <div
      className="overflow-x-auto text-[13px] text-foreground/90 scrollbar-thin"
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}

export function ExplainDrawer({ spec }: { spec: OperationSpec }) {
  const [open, setOpen] = React.useState(false)
  const learn = useUI((s) => s.learnMode)
  if (!learn && !open) {
    return (
      <Button variant="ghost" size="xs" className="w-full justify-start" onClick={() => setOpen(true)}>
        <BookOpen className="h-3 w-3" />
        Explain
      </Button>
    )
  }
  return (
    <div className="space-y-2 rounded border border-border/60 bg-muted/30 p-2.5">
      <div className="flex items-center gap-1.5">
        <Info className="h-3 w-3 text-primary" />
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          How it works
        </span>
      </div>
      {spec.summary && (
        <p className="text-[11px] leading-relaxed text-foreground/85">{spec.summary}</p>
      )}
      {spec.formula_latex && <Formula latex={spec.formula_latex} />}
      {spec.when_to_use && (
        <div>
          <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            When to use
          </span>
          <p className="mt-0.5 text-[11px] leading-relaxed text-foreground/85">{spec.when_to_use}</p>
        </div>
      )}
      {spec.references.length > 0 && (
        <div>
          <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            References
          </span>
          <ul className="mt-0.5 space-y-0.5">
            {spec.references.map((r) => (
              <li key={r} className="text-[10px] leading-snug text-muted-foreground">
                {r}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="flex flex-wrap gap-1 pt-0.5">
        {spec.syllabus && <Badge variant="syllabus">syllabus</Badge>}
        {spec.tags.includes('beyond-syllabus') && <Badge variant="beyond">beyond syllabus</Badge>}
        {spec.domain === 'linear' && (
          <Tooltip content="Runs in linear light, where this operation is physically correct.">
            <Badge variant="outline">linear light</Badge>
          </Tooltip>
        )}
        {spec.cost === 'slow' && <Badge variant="slow">slow</Badge>}
      </div>
    </div>
  )
}

export function OperatorPanel({ op }: { op: StackOp }) {
  const spec = useCatalogue((s) => s.byId[op.op_id])
  const { updateParams, patchOp, commit, preview } = usePipeline()
  const { images, activeId } = useSession()
  const blendModes = useCatalogue((s) => s.catalogue?.blend_modes ?? [])
  const showAdvanced = useUI((s) => s.showAdvanced)
  const [autoBusy, setAutoBusy] = React.useState<string | null>(null)
  const [advOpen, setAdvOpen] = React.useState(false)

  if (!spec) {
    return (
      <div className="p-3 text-xs text-muted-foreground">
        Unknown operator <code>{op.op_id}</code>.
      </div>
    )
  }

  const imageOptions = images
    .filter((i) => i.image_id !== activeId)
    .map((i) => ({ id: i.image_id, label: i.name || i.image_id.slice(0, 8) }))

  const visible = spec.params.filter((p) => paramVisible(p, op.params))
  const basic = visible.filter((p) => !p.advanced)
  const advanced = visible.filter((p) => p.advanced)

  // Histogram behind the curve editor, taken from the live preview.
  const hist = preview?.stats?.histogram

  const runAuto = async (paramName: string) => {
    if (!activeId) return
    setAutoBusy(paramName)
    try {
      const ops = usePipeline.getState().ops
      const index = ops.findIndex((o) => o.uid === op.uid)
      const res = await api.autoParams(spec.id, {
        image_id: activeId,
        ops: wireOpsSafe(index > 0 ? ops.slice(0, index) : []),
      })
      const clean: Record<string, unknown> = {}
      for (const [k, v] of Object.entries(res.params)) {
        if (!k.startsWith('_')) clean[k] = v
      }
      commit()
      updateParams(op.uid, clean)
    } catch {
      /* surfaced by the status bar */
    } finally {
      setAutoBusy(null)
    }
  }

  const renderParam = (p: (typeof spec.params)[number]) => {
    if (p.kind === 'curve') {
      return (
        <CurveEditor
          key={p.name}
          spec={p}
          value={op.params[p.name]}
          onChange={(v) => updateParams(op.uid, { [p.name]: v })}
          onBeginEdit={commit}
          histogram={hist}
        />
      )
    }
    return (
      <ParamControl
        key={p.name}
        spec={p}
        value={op.params[p.name]}
        onChange={(v) => updateParams(op.uid, { [p.name]: v })}
        onBeginEdit={commit}
        onAuto={p.auto ? () => runAuto(p.name) : undefined}
        autoBusy={autoBusy === p.name}
        imageOptions={imageOptions}
      />
    )
  }

  // Group basic params by their declared `group`, preserving order.
  const groups = new Map<string, typeof basic>()
  for (const p of basic) {
    const key = p.group || ''
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)!.push(p)
  }

  return (
    <div className="space-y-3">
      {[...groups.entries()].map(([groupName, params]) => (
        <div key={groupName || 'main'} className="space-y-2.5">
          {groupName && (
            <div className="flex items-center gap-2 pt-1">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                {groupName}
              </span>
              <Separator className="flex-1" />
            </div>
          )}
          {params.map(renderParam)}
        </div>
      ))}

      {advanced.length > 0 && (
        <div className="space-y-2.5">
          <button
            onClick={() => setAdvOpen((v) => !v)}
            className="flex w-full items-center gap-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground"
          >
            <ChevronDown className={cn('h-3 w-3 transition-transform', !advOpen && '-rotate-90')} />
            Advanced ({advanced.length})
          </button>
          {(advOpen || showAdvanced) && advanced.map(renderParam)}
        </div>
      )}

      {!spec.is_view && (
        <>
          <Separator />
          <div className="space-y-2.5">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label>Opacity</Label>
                <span className="tabular text-[11px] text-muted-foreground">
                  {fmtNum(op.opacity * 100, 0)}%
                </span>
              </div>
              <Slider
                value={[op.opacity]}
                min={0}
                max={1}
                step={0.01}
                onPointerDown={commit}
                onValueChange={([v]) => patchOp(op.uid, { opacity: v })}
                aria-label="Operator opacity"
              />
            </div>
            <div className="space-y-1">
              <Label>Blend mode</Label>
              <Select value={op.blend} onValueChange={(v) => patchOp(op.uid, { blend: v })}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {blendModes.map((m) => (
                    <SelectItem key={m.value} value={m.value}>
                      {m.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <MaskPicker op={op} />
          </div>
        </>
      )}

      <Separator />
      <ExplainDrawer spec={spec} />
    </div>
  )
}

// Radix Select reserves the empty string for "no selection" and throws if an
// item uses it, so "no mask" needs an explicit sentinel.
const NO_MASK = 'none'

const MASK_KINDS = [
  { value: NO_MASK, label: 'No mask (whole image)' },
  { value: 'luminosity', label: 'Luminosity range' },
  { value: 'color_range', label: 'Colour range' },
  { value: 'gradient', label: 'Gradient' },
  { value: 'rect', label: 'Rectangle' },
]

function MaskPicker({ op }: { op: StackOp }) {
  const { patchOp, commit } = usePipeline()
  const kind = (op.mask?.kind as string) ?? NO_MASK

  const setKind = (k: string) => {
    commit()
    if (!k || k === NO_MASK) {
      patchOp(op.uid, { mask: null })
      return
    }
    const defaults: Record<string, Record<string, unknown>> = {
      luminosity: { kind: 'luminosity', range: [0.25, 0.75], feather: 0.05 },
      color_range: { kind: 'color_range', hue: [90, 150], sat: [0.15, 1], val: [0.1, 1], feather: 0.03 },
      gradient: { kind: 'gradient', gradient_kind: 'linear', p0: [0, 0], p1: [1, 1] },
      rect: { kind: 'rect', rect: [0.2, 0.2, 0.6, 0.6], feather: 0.03 },
    }
    patchOp(op.uid, { mask: defaults[k] })
  }

  return (
    <div className="space-y-1.5">
      <Tooltip content="Confine this operator to part of the image. The effect is exactly zero where the mask is zero.">
        <Label className="cursor-help">Mask</Label>
      </Tooltip>
      <Select value={kind} onValueChange={setKind}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {MASK_KINDS.map((m) => (
            <SelectItem key={m.value} value={m.value}>
              {m.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {kind === 'luminosity' && (
        <div className="space-y-1.5 rounded border border-border/60 p-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-muted-foreground">Tonal band</span>
            <span className="tabular text-[10px] text-muted-foreground">
              {fmtNum((op.mask?.range as number[])?.[0] ?? 0, 2)} –{' '}
              {fmtNum((op.mask?.range as number[])?.[1] ?? 1, 2)}
            </span>
          </div>
          <Slider
            value={(op.mask?.range as number[]) ?? [0.25, 0.75]}
            min={0}
            max={1}
            step={0.01}
            onPointerDown={commit}
            onValueChange={(v) => patchOp(op.uid, { mask: { ...op.mask, range: v } })}
            aria-label="Luminosity range"
          />
          <p className="text-[10px] leading-snug text-muted-foreground">
            0 is black, 1 is white. Narrow this to the midtones to sharpen
            without touching noise in the shadows.
          </p>
        </div>
      )}

      {kind === 'color_range' && (
        <div className="space-y-1.5 rounded border border-border/60 p-2">
          <span className="text-[10px] text-muted-foreground">
            Hue window (degrees, wraps at 360)
          </span>
          <Slider
            value={(op.mask?.hue as number[]) ?? [90, 150]}
            min={0}
            max={360}
            step={1}
            onPointerDown={commit}
            onValueChange={(v) => patchOp(op.uid, { mask: { ...op.mask, hue: v } })}
            aria-label="Hue window"
          />
          <div className="tabular text-[10px] text-muted-foreground">
            {Math.round((op.mask?.hue as number[])?.[0] ?? 0)}° –{' '}
            {Math.round((op.mask?.hue as number[])?.[1] ?? 360)}°
          </div>
        </div>
      )}

      {kind === 'rect' && (
        <p className="rounded border border-dashed border-border px-2 py-1.5 text-[10px] text-muted-foreground">
          Drag the rectangle on the canvas to position it.
        </p>
      )}
    </div>
  )
}

export function OperatorHeader({ spec }: { spec: OperationSpec }) {
  return (
    <div className="space-y-1">
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-sm font-semibold leading-tight">{spec.name}</h3>
        <div className="flex shrink-0 gap-1">
          {spec.syllabus && <Badge variant="syllabus">S</Badge>}
          {spec.cost === 'slow' && (
            <Tooltip content="Slow at full resolution; the preview runs on the proxy.">
              <Badge variant="slow">slow</Badge>
            </Tooltip>
          )}
        </div>
      </div>
      <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
        <code className="rounded bg-muted px-1 py-0.5">{spec.id}</code>
        <span>{titleCase(spec.category)}</span>
      </div>
    </div>
  )
}
