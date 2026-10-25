/** Studio: the manual editor. Left = operator browser, centre = canvas,
 *  right = inspector (params / histogram / metrics). PLAN.md Section 1. */
import {
  Box, Camera, ChevronRight, Copy, Crosshair, Download, Droplet, FileText,
  GitBranch, Layers, Move, Palette, Radio, Redo2, Search, Shuffle, Sun, Undo2,
  Zap,
} from 'lucide-react'
import * as React from 'react'

import { ImageCanvas } from '@/components/canvas/ImageCanvas'
import { Histogram } from '@/components/charts/Histogram'
import { MetricGauge } from '@/components/charts/ScoreCharts'
import { OperatorHeader, OperatorPanel } from '@/components/controls/OperatorPanel'
import { Dropzone } from '@/components/layout/Shell'
import { OpStack } from '@/components/pipeline/OpStack'
import {
  Badge, Button, EmptyState, Input, Separator, Spinner, Tabs, TabsContent,
  TabsList, TabsTrigger, Tooltip,
} from '@/components/ui'
import { api, waitForJob, wireOps } from '@/lib/api'
import type { OperationSpec } from '@/lib/types'
import { cn, downloadText, downloadUrl, fmtNum, makeOp } from '@/lib/utils'
import { useCatalogue, usePipeline, useSession, useUI } from '@/store'

const CATEGORY_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  inspect: Search, tone: Sun, color: Palette, smooth: Droplet, sharpen: Zap,
  freq: Radio, geom: Move, morph: Box, edge: GitBranch, feat: Crosshair,
  seg: Layers, aug: Shuffle, compose: Copy,
}

export function Studio() {
  const active = useSession((s) => s.images.find((i) => i.image_id === s.activeId) ?? null)
  if (!active) return <Dropzone />
  return (
    <div className="flex min-h-0 flex-1">
      <OperatorBrowser />
      <div className="min-w-0 flex-1">
        <ImageCanvas />
      </div>
      <Inspector />
    </div>
  )
}

function OperatorBrowser() {
  const catalogue = useCatalogue((s) => s.catalogue)
  const { activeCategory, setActiveCategory, search, setSearch } = useUI()
  const { addOp, select } = usePipeline()

  const filtered = React.useMemo(() => {
    if (!catalogue) return []
    const q = search.trim().toLowerCase()
    if (!q) return catalogue.operations.filter((o) => o.category === activeCategory)
    return catalogue.operations.filter(
      (o) =>
        o.name.toLowerCase().includes(q) ||
        o.id.toLowerCase().includes(q) ||
        o.summary.toLowerCase().includes(q) ||
        o.tags.some((t) => t.includes(q)),
    )
  }, [catalogue, activeCategory, search])

  if (!catalogue) {
    return (
      <aside className="flex w-[232px] shrink-0 items-center justify-center border-r border-border bg-card">
        <Spinner className="h-5 w-5" />
      </aside>
    )
  }

  const add = (spec: OperationSpec) => {
    const op = makeOp(spec.id, spec.params)
    addOp(op)
    select(op.uid)
  }

  return (
    <aside className="flex w-[232px] shrink-0 flex-col border-r border-border bg-card">
      <div className="shrink-0 border-b border-border p-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={`Search ${catalogue.total} operators...`}
            className="pl-6.5"
            style={{ paddingLeft: 24 }}
            aria-label="Search operators"
          />
        </div>
      </div>

      {!search && (
        <div className="shrink-0 overflow-x-auto border-b border-border p-1 scrollbar-thin">
          <div className="flex flex-wrap gap-0.5">
            {catalogue.categories.map((c) => {
              const Icon = CATEGORY_ICONS[c.id] ?? Layers
              const activeCat = activeCategory === c.id
              return (
                <Tooltip key={c.id} content={c.blurb}>
                  <button
                    onClick={() => setActiveCategory(c.id)}
                    className={cn(
                      'flex items-center gap-1 rounded px-1.5 py-1 text-[10px] transition-colors',
                      activeCat
                        ? 'bg-primary/15 text-primary'
                        : 'text-muted-foreground hover:bg-accent hover:text-foreground',
                    )}
                  >
                    <Icon className="h-3 w-3" />
                    {c.label}
                    <span className="opacity-50">{c.count}</span>
                  </button>
                </Tooltip>
              )
            })}
          </div>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto p-1 scrollbar-thin">
        {filtered.length === 0 && (
          <EmptyState title="No matches" className="py-6">
            Try a different search term.
          </EmptyState>
        )}
        {filtered.map((spec) => (
          <button
            key={spec.id}
            onClick={() => add(spec)}
            className="group flex w-full items-start gap-1.5 rounded px-1.5 py-1.5 text-left transition-colors hover:bg-accent"
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1">
                <span className="truncate text-xs font-medium">{spec.name}</span>
                {spec.syllabus && <Badge variant="syllabus">S</Badge>}
                {spec.is_view && <Badge variant="outline">view</Badge>}
                {spec.cost === 'slow' && <Badge variant="slow">slow</Badge>}
              </div>
              {spec.summary && (
                <p className="mt-0.5 line-clamp-2 text-[10px] leading-snug text-muted-foreground">
                  {spec.summary}
                </p>
              )}
            </div>
            <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
          </button>
        ))}
      </div>

      <div className="shrink-0 border-t border-border px-2 py-1.5">
        <p className="text-[9px] leading-snug text-muted-foreground">
          {catalogue.total} operators · {catalogue.syllabus_count} syllabus topics ·
          no learned models
        </p>
      </div>
    </aside>
  )
}

function Inspector() {
  const { ops, selectedUid, preview, undo, redo, past, future, addSnapshot, snapshots, restoreSnapshot } = usePipeline()
  const byId = useCatalogue((s) => s.byId)
  const { inspectorTab, setInspectorTab } = useUI()
  const activeId = useSession((s) => s.activeId)
  const selected = ops.find((o) => o.uid === selectedUid)
  const spec = selected ? byId[selected.op_id] : undefined

  const viewData = React.useMemo(() => {
    if (!preview?.data) return []
    return Object.entries(preview.data)
  }, [preview])

  return (
    <aside className="flex w-[306px] shrink-0 flex-col border-l border-border bg-card">
      <div className="flex h-9 shrink-0 items-center gap-1 border-b border-border px-2">
        <Tooltip content="Undo (ctrl+Z)">
          <Button variant="ghost" size="icon-sm" onClick={undo} disabled={!past.length} aria-label="Undo">
            <Undo2 className="h-3.5 w-3.5" />
          </Button>
        </Tooltip>
        <Tooltip content="Redo (ctrl+shift+Z)">
          <Button variant="ghost" size="icon-sm" onClick={redo} disabled={!future.length} aria-label="Redo">
            <Redo2 className="h-3.5 w-3.5" />
          </Button>
        </Tooltip>
        <Tooltip content="Snapshot the current stack for comparison">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => addSnapshot(`Snapshot ${snapshots.length + 1}`, preview?.render.url)}
            aria-label="Take snapshot"
          >
            <Camera className="h-3.5 w-3.5" />
          </Button>
        </Tooltip>
        <div className="flex-1" />
        <ExportMenu />
      </div>

      <div className="max-h-[38%] shrink-0 overflow-y-auto border-b border-border p-2 scrollbar-thin">
        <OpStack />
        {snapshots.length > 0 && (
          <div className="mt-2 space-y-1">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Snapshots
            </span>
            {snapshots.map((s) => (
              <button
                key={s.id}
                onClick={() => restoreSnapshot(s.id)}
                className="flex w-full items-center gap-1.5 rounded px-1 py-0.5 text-left hover:bg-accent"
              >
                {s.renderUrl && (
                  <img src={s.renderUrl} alt="" className="h-6 w-8 rounded object-cover" />
                )}
                <span className="truncate text-[10px]">{s.name}</span>
                <span className="flex-1" />
                <span className="text-[9px] text-muted-foreground">{s.ops.length} ops</span>
              </button>
            ))}
          </div>
        )}
      </div>

      <Tabs
        value={inspectorTab}
        onValueChange={(v) => setInspectorTab(v as 'params' | 'histogram' | 'metrics')}
        className="flex min-h-0 flex-1 flex-col"
      >
        <TabsList className="mx-2 mt-2 shrink-0 self-start">
          <TabsTrigger value="params">Controls</TabsTrigger>
          <TabsTrigger value="histogram">Histogram</TabsTrigger>
          <TabsTrigger value="metrics">Readouts</TabsTrigger>
        </TabsList>

        <div className="min-h-0 flex-1 overflow-y-auto p-2 scrollbar-thin">
          <TabsContent value="params" className="mt-0 space-y-3">
            {!selected || !spec ? (
              <EmptyState title="No operator selected" className="py-8">
                Select an operator in the pipeline above, or add one from the
                left, to edit its parameters here.
              </EmptyState>
            ) : (
              <>
                <OperatorHeader spec={spec} />
                <Separator />
                <OperatorPanel op={selected} />
              </>
            )}
          </TabsContent>

          <TabsContent value="histogram" className="mt-0 space-y-3">
            <Histogram stats={preview?.stats} height={120} />
            {preview?.stats && (
              <div className="space-y-2">
                <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Against expected ranges
                </span>
                <MetricGauge
                  label="mean luminance"
                  value={preview.stats.mean}
                  expected={[0.35, 0.6]}
                  help="A well-exposed image averages between 0.35 and 0.60."
                />
                <MetricGauge
                  label="RMS contrast"
                  value={preview.stats.std}
                  expected={[0.1, 0.3]}
                  help="Standard deviation of luminance."
                />
                <MetricGauge
                  label="highlight clipping"
                  value={preview.stats.clip_high}
                  expected={[0, 0.005]}
                  help="Fraction of pixels at or above 253/255, where detail is lost."
                />
                <MetricGauge
                  label="shadow clipping"
                  value={preview.stats.clip_low}
                  expected={[0, 0.01]}
                  help="Fraction of pixels at or below 2/255."
                />
              </div>
            )}
          </TabsContent>

          <TabsContent value="metrics" className="mt-0 space-y-3">
            {viewData.length === 0 ? (
              <EmptyState title="No readouts" className="py-8">
                Operators that measure rather than modify report their numbers
                here. Try adding something from the Inspect category.
              </EmptyState>
            ) : (
              viewData.map(([key, data]) => (
                <div key={key} className="space-y-1">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                    {byId[key.split('#')[0]]?.name ?? key}
                  </span>
                  <div className="space-y-0.5 rounded border border-border/60 bg-muted/30 p-2">
                    {Object.entries(data).map(([k, v]) => (
                      <ReadoutRow key={k} name={k} value={v} />
                    ))}
                  </div>
                </div>
              ))
            )}
          </TabsContent>
        </div>
      </Tabs>

      {activeId && (
        <div className="shrink-0 border-t border-border p-2">
          <ReportButton />
        </div>
      )}
    </aside>
  )
}

function ReadoutRow({ name, value }: { name: string; value: unknown }) {
  const [open, setOpen] = React.useState(false)
  if (Array.isArray(value)) {
    const preview = value.slice(0, 4).map((v) => (typeof v === 'number' ? fmtNum(v, 3) : String(v)))
    const long = value.length > 4
    return (
      <div className="text-[10px]">
        <button
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-baseline justify-between gap-2 text-left"
        >
          <span className="text-muted-foreground">{name.replace(/_/g, ' ')}</span>
          <span className="tabular">
            [{preview.join(', ')}
            {long ? `, +${value.length - 4}` : ''}]
          </span>
        </button>
        {open && long && (
          <pre className="mt-1 max-h-32 overflow-auto rounded bg-background/60 p-1 text-[9px] scrollbar-thin">
            {JSON.stringify(value, null, 1)}
          </pre>
        )}
      </div>
    )
  }
  if (value !== null && typeof value === 'object') {
    return (
      <div className="text-[10px]">
        <button onClick={() => setOpen((v) => !v)} className="w-full text-left text-muted-foreground">
          {name.replace(/_/g, ' ')} {open ? '▾' : '▸'}
        </button>
        {open && (
          <pre className="mt-1 max-h-40 overflow-auto rounded bg-background/60 p-1 text-[9px] scrollbar-thin">
            {JSON.stringify(value, null, 1)}
          </pre>
        )}
      </div>
    )
  }
  return (
    <div className="flex items-baseline justify-between gap-2 text-[10px]">
      <span className="text-muted-foreground">{name.replace(/_/g, ' ')}</span>
      <span className="tabular">
        {typeof value === 'number' ? fmtNum(value, 5) : String(value)}
      </span>
    </div>
  )
}

function ExportMenu() {
  const activeId = useSession((s) => s.activeId)
  const ops = usePipeline((s) => s.ops)
  const [busy, setBusy] = React.useState(false)
  const [progress, setProgress] = React.useState(0)

  const doExport = async (format: string, bitDepth = 8) => {
    if (!activeId) return
    setBusy(true)
    setProgress(0)
    try {
      const { job_id } = await api.export({
        image_id: activeId,
        ops: wireOps(ops),
        format,
        quality: 95,
        bit_depth: bitDepth,
      })
      const job = await waitForJob(job_id, (j) => setProgress(j.progress))
      if (job.status === 'done') {
        const r = job.result as { url: string; filename: string }
        downloadUrl(r.filename, r.url)
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex items-center gap-0.5">
      <Tooltip content="Export at full resolution as PNG">
        <Button variant="ghost" size="icon-sm" onClick={() => doExport('png')} disabled={busy || !activeId} aria-label="Export PNG">
          {busy ? <Spinner /> : <Download className="h-3.5 w-3.5" />}
        </Button>
      </Tooltip>
      <Tooltip content="Export as JPEG (quality 95)">
        <Button variant="ghost" size="xs" onClick={() => doExport('jpeg')} disabled={busy || !activeId}>
          JPG
        </Button>
      </Tooltip>
      <Tooltip content="Export as 16-bit PNG">
        <Button variant="ghost" size="xs" onClick={() => doExport('png', 16)} disabled={busy || !activeId}>
          16b
        </Button>
      </Tooltip>
      {busy && progress > 0 && (
        <span className="text-[9px] tabular text-muted-foreground">{Math.round(progress * 100)}%</span>
      )}
    </div>
  )
}

function ReportButton() {
  const activeId = useSession((s) => s.activeId)
  const ops = usePipeline((s) => s.ops)
  const [busy, setBusy] = React.useState(false)

  const generate = async () => {
    if (!activeId) return
    setBusy(true)
    try {
      const r = await api.report({
        image_id: activeId,
        ops: wireOps(ops),
        include_advisor: true,
        include_images: true,
      })
      downloadText('image-processing-report.md', r.markdown, 'text/markdown')
    } finally {
      setBusy(false)
    }
  }

  return (
    <Tooltip content="Generate a markdown report of this session: the recipe, each operator's theory, measured before/after metrics, and the Advisor's findings.">
      <Button variant="outline" size="sm" className="w-full" onClick={generate} disabled={busy}>
        {busy ? <Spinner /> : <FileText className="h-3.5 w-3.5" />}
        Generate lab report
      </Button>
    </Tooltip>
  )
}
