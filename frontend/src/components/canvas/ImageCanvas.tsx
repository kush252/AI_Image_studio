/** The main viewport: zoom/pan, before/after comparison, and interactive
 *  parameter overlays (points, rects, lines) mapped through the transform. */
import {
  Columns2, Crosshair, Eye, Maximize2, Minus, Plus, SplitSquareHorizontal,
  SquareStack,
} from 'lucide-react'
import * as React from 'react'

import { Badge, Button, Tooltip } from '@/components/ui'
import type { ParamSpec, StackOp } from '@/lib/types'
import { clamp, cn } from '@/lib/utils'
import { useCatalogue, usePipeline, useSession, useUI } from '@/store'

interface Transform {
  scale: number
  x: number
  y: number
}

export function ImageCanvas() {
  const containerRef = React.useRef<HTMLDivElement>(null)
  const { preview, previewing, ops, selectedUid, updateParams, commit } = usePipeline()
  const active = useSession((s) => s.images.find((i) => i.image_id === s.activeId) ?? null)
  const {
    compare, splitPosition, setSplitPosition, holdBefore, zoom, fit, pan,
    setZoom, setFit, setPan, resetView, showOverlay,
  } = useUI()
  const byId = useCatalogue((s) => s.byId)

  const [size, setSize] = React.useState({ w: 0, h: 0 })
  const [dragging, setDragging] = React.useState<null | 'pan' | 'split' | string>(null)
  const [cursorValue, setCursorValue] = React.useState<{ x: number; y: number } | null>(null)

  React.useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      setSize({ w: el.clientWidth, h: el.clientHeight })
    })
    ro.observe(el)
    setSize({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  const imgW = preview?.width ?? active?.proxy_width ?? 1
  const imgH = preview?.height ?? active?.proxy_height ?? 1

  // Fit scale leaves a small margin so the frame edge is visible.
  const fitScale = React.useMemo(() => {
    if (!size.w || !size.h) return 1
    return Math.min((size.w - 32) / imgW, (size.h - 32) / imgH)
  }, [size, imgW, imgH])

  const t: Transform = React.useMemo(() => {
    const scale = fit ? fitScale : zoom
    return {
      scale,
      x: size.w / 2 - (imgW * scale) / 2 + (fit ? 0 : pan.x),
      y: size.h / 2 - (imgH * scale) / 2 + (fit ? 0 : pan.y),
    }
  }, [fit, fitScale, zoom, size, imgW, imgH, pan])

  const beforeUrl = active?.preview?.url
  const afterUrl = preview?.render.url ?? beforeUrl
  const overlayUrl = showOverlay ? preview?.overlays?.[showOverlay]?.url : undefined
  const displayUrl = holdBefore ? beforeUrl : (overlayUrl ?? afterUrl)

  /* ------------------------------------------------------- interactions */

  const toImageCoords = (clientX: number, clientY: number) => {
    const rect = containerRef.current!.getBoundingClientRect()
    return {
      x: clamp((clientX - rect.left - t.x) / (imgW * t.scale), 0, 1),
      y: clamp((clientY - rect.top - t.y) / (imgH * t.scale), 0, 1),
    }
  }

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    const rect = containerRef.current!.getBoundingClientRect()
    const cx = e.clientX - rect.left
    const cy = e.clientY - rect.top
    const current = fit ? fitScale : zoom
    const factor = Math.exp(-e.deltaY * 0.0016)
    const next = clamp(current * factor, 0.05, 32)
    // Keep the point under the cursor fixed while zooming.
    const ix = (cx - t.x) / current
    const iy = (cy - t.y) / current
    const nx = cx - ix * next - (size.w / 2 - (imgW * next) / 2)
    const ny = cy - iy * next - (size.h / 2 - (imgH * next) / 2)
    setZoom(next)
    setPan({ x: nx, y: ny })
  }

  const selectedOp = ops.find((o) => o.uid === selectedUid)
  const selectedSpec = selectedOp ? byId[selectedOp.op_id] : undefined
  const interactiveParams = (selectedSpec?.params ?? []).filter((p) =>
    ['point', 'points', 'rect', 'line'].includes(p.kind),
  )

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button === 1 || e.altKey || !interactiveParams.length) {
      setDragging('pan')
      return
    }
    setDragging('pan')
  }

  React.useEffect(() => {
    if (!dragging) return
    const move = (e: PointerEvent) => {
      if (dragging === 'pan') {
        setFit(false)
        setPan({ x: pan.x + e.movementX, y: pan.y + e.movementY })
      } else if (dragging === 'split') {
        const rect = containerRef.current!.getBoundingClientRect()
        setSplitPosition((e.clientX - rect.left) / rect.width)
      } else if (dragging.startsWith('handle:')) {
        const [, paramName, idxStr] = dragging.split(':')
        if (!selectedOp) return
        const p = toImageCoords(e.clientX, e.clientY)
        const spec = interactiveParams.find((ip) => ip.name === paramName)
        if (!spec) return
        if (spec.kind === 'point') {
          updateParams(selectedOp.uid, { [paramName]: [p.x, p.y] })
        } else if (spec.kind === 'points') {
          const arr = [...((selectedOp.params[paramName] as number[][]) ?? [])]
          arr[Number(idxStr)] = [p.x, p.y]
          updateParams(selectedOp.uid, { [paramName]: arr })
        } else if (spec.kind === 'line') {
          const arr = [...((selectedOp.params[paramName] as number[][]) ?? [[0, 0], [1, 1]])]
          arr[Number(idxStr)] = [p.x, p.y]
          updateParams(selectedOp.uid, { [paramName]: arr })
        } else if (spec.kind === 'rect') {
          const r = [...((selectedOp.params[paramName] as number[]) ?? [0, 0, 1, 1])]
          const corner = Number(idxStr)
          if (corner === 0) {
            const nx = Math.min(p.x, r[0] + r[2] - 0.01)
            const ny = Math.min(p.y, r[1] + r[3] - 0.01)
            updateParams(selectedOp.uid, {
              [paramName]: [nx, ny, r[0] + r[2] - nx, r[1] + r[3] - ny],
            })
          } else {
            updateParams(selectedOp.uid, {
              [paramName]: [r[0], r[1], Math.max(p.x - r[0], 0.01), Math.max(p.y - r[1], 0.01)],
            })
          }
        }
      }
    }
    const up = () => setDragging(null)
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [dragging, pan, selectedOp, interactiveParams]) // eslint-disable-line react-hooks/exhaustive-deps

  const imgStyle: React.CSSProperties = {
    width: imgW * t.scale,
    height: imgH * t.scale,
    transform: `translate(${t.x}px, ${t.y}px)`,
    imageRendering: t.scale >= 3 ? 'pixelated' : 'auto',
  }

  if (!active) return null

  return (
    <div className="relative flex h-full flex-col">
      <CanvasToolbar />

      <div
        ref={containerRef}
        className="relative flex-1 select-none overflow-hidden bg-[hsl(220_18%_7%)]"
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={(e) => {
          if (!dragging) setCursorValue(toImageCoords(e.clientX, e.clientY))
        }}
        onPointerLeave={() => setCursorValue(null)}
        onDoubleClick={resetView}
        style={{ cursor: dragging === 'pan' ? 'grabbing' : 'grab' }}
        role="img"
        aria-label="Image viewport"
      >
        {/* Side-by-side */}
        {compare === 'side' ? (
          <div className="absolute inset-0 flex">
            <div className="relative flex-1 overflow-hidden border-r border-border">
              <img src={beforeUrl} alt="Original" className="absolute origin-top-left alpha-grid" style={imgStyle} draggable={false} />
              <Badge className="absolute left-2 top-2" variant="secondary">Original</Badge>
            </div>
            <div className="relative flex-1 overflow-hidden">
              <img src={afterUrl} alt="Processed" className="absolute origin-top-left alpha-grid" style={imgStyle} draggable={false} />
              <Badge className="absolute left-2 top-2" variant="default">Processed</Badge>
            </div>
          </div>
        ) : compare === 'split' ? (
          <>
            <img src={afterUrl} alt="Processed" className="absolute origin-top-left alpha-grid" style={imgStyle} draggable={false} />
            <div
              className="absolute inset-0 overflow-hidden"
              style={{ clipPath: `inset(0 ${(1 - splitPosition) * 100}% 0 0)` }}
            >
              <img src={beforeUrl} alt="Original" className="absolute origin-top-left alpha-grid" style={imgStyle} draggable={false} />
            </div>
            <div
              className="absolute top-0 z-10 h-full w-0.5 cursor-ew-resize bg-primary"
              style={{ left: `${splitPosition * 100}%` }}
              onPointerDown={(e) => {
                e.stopPropagation()
                setDragging('split')
              }}
            >
              <div className="absolute left-1/2 top-1/2 flex h-7 w-7 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-primary bg-background">
                <SplitSquareHorizontal className="h-3.5 w-3.5 text-primary" />
              </div>
            </div>
            <Badge className="absolute left-2 top-2" variant="secondary">Original</Badge>
            <Badge className="absolute right-2 top-2" variant="default">Processed</Badge>
          </>
        ) : (
          <img
            src={displayUrl}
            alt={holdBefore ? 'Original' : 'Processed'}
            className="absolute origin-top-left alpha-grid"
            style={imgStyle}
            draggable={false}
          />
        )}

        {/* Interactive handles for the selected operator's canvas params */}
        {compare === 'single' && selectedOp && interactiveParams.length > 0 && (
          <svg
            className="pointer-events-none absolute inset-0"
            width={size.w}
            height={size.h}
            style={{ overflow: 'visible' }}
          >
            {interactiveParams.map((p) => (
              <Handles
                key={p.name}
                spec={p}
                op={selectedOp}
                t={t}
                imgW={imgW}
                imgH={imgH}
                onGrab={(name, idx) => {
                  commit()
                  setDragging(`handle:${name}:${idx}`)
                }}
              />
            ))}
          </svg>
        )}

        {previewing && (
          <div className="absolute right-3 top-3 z-20 flex items-center gap-1.5 rounded bg-background/85 px-2 py-1 text-[11px] backdrop-blur">
            <span className="inline-block h-2.5 w-2.5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            Rendering
          </div>
        )}

        {holdBefore && (
          <Badge className="absolute bottom-3 left-1/2 -translate-x-1/2" variant="secondary">
            Showing original (hold space)
          </Badge>
        )}

        {showOverlay && (
          <Badge className="absolute bottom-3 left-3" variant="default">
            Overlay: {showOverlay}
          </Badge>
        )}

        {cursorValue && (
          <div className="pointer-events-none absolute bottom-3 right-3 rounded bg-background/85 px-2 py-1 text-[10px] tabular backdrop-blur">
            x {Math.round(cursorValue.x * imgW)} · y {Math.round(cursorValue.y * imgH)}
          </div>
        )}
      </div>
    </div>
  )
}

function Handles({
  spec, op, t, imgW, imgH, onGrab,
}: {
  spec: ParamSpec
  op: StackOp
  t: Transform
  imgW: number
  imgH: number
  onGrab: (name: string, idx: number) => void
}) {
  const toPx = (nx: number, ny: number) => ({
    cx: t.x + nx * imgW * t.scale,
    cy: t.y + ny * imgH * t.scale,
  })

  const handle = (nx: number, ny: number, idx: number, key: string) => {
    const { cx, cy } = toPx(nx, ny)
    return (
      <circle
        key={key}
        cx={cx}
        cy={cy}
        r={7}
        fill="hsl(var(--background))"
        stroke="hsl(var(--primary))"
        strokeWidth={2.5}
        className="pointer-events-auto cursor-grab"
        onPointerDown={(e) => {
          e.stopPropagation()
          onGrab(spec.name, idx)
        }}
      />
    )
  }

  if (spec.kind === 'point') {
    const p = (op.params[spec.name] as number[]) ?? [0.5, 0.5]
    const { cx, cy } = toPx(p[0], p[1])
    return (
      <g>
        <line x1={cx - 14} y1={cy} x2={cx + 14} y2={cy} stroke="hsl(var(--primary))" strokeWidth={1} opacity={0.6} />
        <line x1={cx} y1={cy - 14} x2={cx} y2={cy + 14} stroke="hsl(var(--primary))" strokeWidth={1} opacity={0.6} />
        {handle(p[0], p[1], 0, 'pt')}
      </g>
    )
  }

  if (spec.kind === 'points') {
    const pts = (op.params[spec.name] as number[][]) ?? []
    const path = pts.map((p) => toPx(p[0], p[1]))
    return (
      <g>
        {path.length > 1 && (
          <polygon
            points={path.map((p) => `${p.cx},${p.cy}`).join(' ')}
            fill="hsl(var(--primary) / 0.08)"
            stroke="hsl(var(--primary))"
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
        )}
        {pts.map((p, i) => handle(p[0], p[1], i, `p${i}`))}
      </g>
    )
  }

  if (spec.kind === 'line') {
    const l = (op.params[spec.name] as number[][]) ?? [[0.1, 0.5], [0.9, 0.5]]
    const a = toPx(l[0][0], l[0][1])
    const b = toPx(l[1][0], l[1][1])
    return (
      <g>
        <line x1={a.cx} y1={a.cy} x2={b.cx} y2={b.cy} stroke="hsl(var(--primary))" strokeWidth={2} />
        {handle(l[0][0], l[0][1], 0, 'l0')}
        {handle(l[1][0], l[1][1], 1, 'l1')}
      </g>
    )
  }

  if (spec.kind === 'rect') {
    const r = (op.params[spec.name] as number[]) ?? [0, 0, 1, 1]
    const a = toPx(r[0], r[1])
    const b = toPx(r[0] + r[2], r[1] + r[3])
    return (
      <g>
        <rect
          x={Math.min(a.cx, b.cx)}
          y={Math.min(a.cy, b.cy)}
          width={Math.abs(b.cx - a.cx)}
          height={Math.abs(b.cy - a.cy)}
          fill="hsl(var(--primary) / 0.08)"
          stroke="hsl(var(--primary))"
          strokeWidth={1.5}
        />
        {handle(r[0], r[1], 0, 'r0')}
        {handle(r[0] + r[2], r[1] + r[3], 1, 'r1')}
      </g>
    )
  }
  return null
}

function CanvasToolbar() {
  const { compare, setCompare, zoom, fit, setZoom, setFit, resetView, showOverlay, setShowOverlay } = useUI()
  const preview = usePipeline((s) => s.preview)
  const overlayNames = Object.keys(preview?.overlays ?? {})
  const scale = fit ? null : zoom

  return (
    <div className="flex h-9 shrink-0 items-center gap-1 border-b border-border bg-card px-2">
      <div className="flex items-center gap-0.5">
        {([
          ['single', Eye, 'Processed only'],
          ['split', SplitSquareHorizontal, 'Split slider'],
          ['side', Columns2, 'Side by side'],
        ] as const).map(([mode, Icon, label]) => (
          <Tooltip key={mode} content={label}>
            <Button
              variant={compare === mode ? 'secondary' : 'ghost'}
              size="icon-sm"
              onClick={() => setCompare(mode)}
              aria-label={label}
              aria-pressed={compare === mode}
            >
              <Icon className="h-3.5 w-3.5" />
            </Button>
          </Tooltip>
        ))}
      </div>

      <div className="mx-1 h-5 w-px bg-border" />

      <Tooltip content="Zoom out">
        <Button variant="ghost" size="icon-sm" onClick={() => setZoom((scale ?? 1) / 1.4)} aria-label="Zoom out">
          <Minus className="h-3.5 w-3.5" />
        </Button>
      </Tooltip>
      <button
        onClick={() => (fit ? setZoom(1) : setFit(true))}
        className="min-w-[52px] rounded px-1.5 py-0.5 text-[11px] tabular text-muted-foreground hover:bg-accent hover:text-foreground"
      >
        {fit ? 'Fit' : `${Math.round(zoom * 100)}%`}
      </button>
      <Tooltip content="Zoom in">
        <Button variant="ghost" size="icon-sm" onClick={() => setZoom((scale ?? 1) * 1.4)} aria-label="Zoom in">
          <Plus className="h-3.5 w-3.5" />
        </Button>
      </Tooltip>
      <Tooltip content="Fit to window (double-click the canvas)">
        <Button variant="ghost" size="icon-sm" onClick={resetView} aria-label="Fit to window">
          <Maximize2 className="h-3.5 w-3.5" />
        </Button>
      </Tooltip>
      <Tooltip content="100% (1 image pixel per screen pixel)">
        <Button variant="ghost" size="icon-sm" onClick={() => setZoom(1)} aria-label="Actual size">
          <Crosshair className="h-3.5 w-3.5" />
        </Button>
      </Tooltip>

      {overlayNames.length > 0 && (
        <>
          <div className="mx-1 h-5 w-px bg-border" />
          <SquareStack className="h-3.5 w-3.5 text-muted-foreground" />
          <div className="flex items-center gap-0.5 overflow-x-auto scrollbar-thin">
            {overlayNames.map((n) => (
              <button
                key={n}
                onClick={() => setShowOverlay(showOverlay === n ? null : n)}
                className={cn(
                  'whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] transition-colors',
                  showOverlay === n
                    ? 'bg-primary/20 text-primary'
                    : 'text-muted-foreground hover:bg-accent',
                )}
              >
                {n.split(':').pop()}
              </button>
            ))}
          </div>
        </>
      )}

      <div className="flex-1" />

      <span className="text-[10px] text-muted-foreground">hold space for original</span>
    </div>
  )
}
