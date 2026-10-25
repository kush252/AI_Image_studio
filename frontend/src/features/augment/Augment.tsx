/** Augmentation Lab: build a recipe of randomised stages, sample a gallery of
 *  variants, and batch-generate a reproducible dataset. PLAN.md Phase 7. */
import {
  Dices, Download, Layers, Package, Pin, Plus, RefreshCw, X,
} from 'lucide-react'
import * as React from 'react'

import { Dropzone } from '@/components/layout/Shell'
import { ParamControl } from '@/components/controls/ParamControl'
import {
  Badge, Button, EmptyState, Input, Label, Select, SelectContent, SelectItem,
  SelectTrigger, SelectValue, Slider, Spinner, Tooltip,
} from '@/components/ui'
import { api, waitForJob } from '@/lib/api'
import type { AugPreset, AugStage, AugVariant } from '@/lib/types'
import { cn, downloadUrl, fmtNum, paramVisible, uid } from '@/lib/utils'
import { useCatalogue, usePipeline, useSession } from '@/store'

interface StageRow extends AugStage {
  rowId: string
}

export function Augment() {
  const active = useSession((s) => s.images.find((i) => i.image_id === s.activeId) ?? null)
  const images = useSession((s) => s.images)
  const catalogue = useCatalogue((s) => s.catalogue)
  const byId = useCatalogue((s) => s.byId)
  const { setOps, commit } = usePipeline()

  const [stages, setStages] = React.useState<StageRow[]>([])
  const [variants, setVariants] = React.useState<AugVariant[]>([])
  const [count, setCount] = React.useState(8)
  const [seed, setSeed] = React.useState(1234)
  const [busy, setBusy] = React.useState(false)
  const [presets, setPresets] = React.useState<AugPreset[]>([])
  const [selectedStage, setSelectedStage] = React.useState<string | null>(null)
  const [pinned, setPinned] = React.useState<number[]>([])
  const [gen, setGen] = React.useState<{ progress: number; message: string } | null>(null)
  const [variantsPerImage, setVariantsPerImage] = React.useState(5)
  const [selectedImages, setSelectedImages] = React.useState<string[]>([])
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    api.augPresets().then((r) => setPresets(r.presets)).catch(() => undefined)
  }, [])

  React.useEffect(() => {
    if (active && !selectedImages.length) setSelectedImages([active.image_id])
  }, [active]) // eslint-disable-line react-hooks/exhaustive-deps

  const augOps = React.useMemo(
    () => (catalogue?.operations ?? []).filter((o) => o.category === 'aug'),
    [catalogue],
  )

  const sample = React.useCallback(async () => {
    if (!active || !stages.length) {
      setVariants([])
      return
    }
    setBusy(true)
    setError(null)
    try {
      const r = await api.augPreview({
        image_id: active.image_id,
        stages: stages.map((s) => ({ op_id: s.op_id, params: s.params, p: s.p })),
        count,
        seed,
      })
      setVariants(r.variants)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [active, stages, count, seed])

  React.useEffect(() => {
    void sample()
  }, [sample])

  if (!active) return <Dropzone />

  const addStage = (opId: string) => {
    const spec = byId[opId]
    if (!spec) return
    const params: Record<string, unknown> = {}
    for (const p of spec.params) {
      if (p.name !== 'p') params[p.name] = p.default
    }
    const row: StageRow = { rowId: uid('stage'), op_id: opId, params, p: 1 }
    setStages((s) => [...s, row])
    setSelectedStage(row.rowId)
  }

  const applyPreset = (preset: AugPreset) => {
    setStages(
      preset.stages.map((s) => {
        const spec = byId[s.op_id]
        const params: Record<string, unknown> = {}
        for (const p of spec?.params ?? []) {
          if (p.name !== 'p') params[p.name] = p.default
        }
        return { rowId: uid('stage'), op_id: s.op_id, params: { ...params, ...s.params }, p: s.p }
      }),
    )
  }

  const generate = async () => {
    if (!selectedImages.length || !stages.length) return
    setGen({ progress: 0, message: 'starting' })
    setError(null)
    try {
      const { job_id } = await api.augGenerate({
        image_ids: selectedImages,
        stages: stages.map((s) => ({ op_id: s.op_id, params: s.params, p: s.p })),
        variants_per_image: variantsPerImage,
        seed,
        format: 'jpeg',
        quality: 92,
      })
      const job = await waitForJob(job_id, (j) =>
        setGen({ progress: j.progress, message: j.message }),
      )
      if (job.status === 'done') {
        const r = job.result as { url: string; count: number }
        downloadUrl('augmented-dataset.zip', r.url)
      } else if (job.status === 'error') {
        setError(job.error ?? 'Generation failed')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setGen(null)
    }
  }

  const promote = (v: AugVariant) => {
    commit()
    setOps(
      v.recipe.map((o) => ({
        uid: uid('aug'),
        op_id: o.op_id,
        params: { ...o.params },
        enabled: true,
        opacity: 1,
        blend: 'normal',
        mask: null,
      })),
      false,
    )
  }

  const stage = stages.find((s) => s.rowId === selectedStage)
  const stageSpec = stage ? byId[stage.op_id] : undefined

  return (
    <div className="flex min-h-0 flex-1">
      {/* ------------------------------------------------- recipe column */}
      <aside className="flex w-[280px] shrink-0 flex-col border-r border-border bg-card">
        <div className="shrink-0 space-y-2 border-b border-border p-2">
          <div className="flex items-center gap-1">
            <Layers className="h-3.5 w-3.5 text-primary" />
            <span className="text-xs font-semibold">Recipe</span>
            <div className="flex-1" />
            <span className="text-[10px] text-muted-foreground">{stages.length} stages</span>
          </div>

          <Select value="" onValueChange={addStage}>
            <SelectTrigger>
              <SelectValue placeholder="Add a stage..." />
            </SelectTrigger>
            <SelectContent>
              {augOps.map((o) => (
                <SelectItem key={o.id} value={o.id}>
                  {o.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value="" onValueChange={(v) => {
            const p = presets.find((x) => x.id === v)
            if (p) applyPreset(p)
          }}>
            <SelectTrigger>
              <SelectValue placeholder="Load a preset..." />
            </SelectTrigger>
            <SelectContent>
              {presets.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-2 scrollbar-thin">
          {!stages.length ? (
            <EmptyState title="No stages" className="py-6">
              Add stages, or load a preset. Each stage declares parameter RANGES
              and a probability, so the recipe is a distribution rather than a
              fixed edit.
            </EmptyState>
          ) : (
            <div className="space-y-1">
              {stages.map((s, i) => (
                <div
                  key={s.rowId}
                  className={cn(
                    'rounded border px-1.5 py-1 transition-colors',
                    selectedStage === s.rowId
                      ? 'border-primary/60 bg-primary/10'
                      : 'border-border/60 hover:bg-accent/40',
                  )}
                >
                  <div className="flex items-center gap-1">
                    <span className="w-3 text-[10px] tabular text-muted-foreground">{i + 1}</span>
                    <button
                      onClick={() => setSelectedStage(s.rowId === selectedStage ? null : s.rowId)}
                      className="min-w-0 flex-1 truncate text-left text-xs font-medium"
                    >
                      {byId[s.op_id]?.name ?? s.op_id}
                    </button>
                    <span className="text-[9px] tabular text-muted-foreground">
                      p={s.p.toFixed(2)}
                    </span>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => setStages((x) => x.filter((y) => y.rowId !== s.rowId))}
                      aria-label="Remove stage"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                  <Slider
                    value={[s.p]}
                    min={0}
                    max={1}
                    step={0.05}
                    onValueChange={([v]) =>
                      setStages((x) => x.map((y) => (y.rowId === s.rowId ? { ...y, p: v } : y)))
                    }
                    className="mt-1"
                    aria-label="Stage probability"
                  />
                </div>
              ))}
            </div>
          )}

          {stage && stageSpec && (
            <div className="mt-3 space-y-2.5 rounded border border-border/60 bg-muted/20 p-2">
              <div className="flex items-center gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  {stageSpec.name}
                </span>
                {stageSpec.syllabus && <Badge variant="syllabus">S</Badge>}
              </div>
              {stageSpec.summary && (
                <p className="text-[10px] leading-snug text-muted-foreground">
                  {stageSpec.summary}
                </p>
              )}
              {stageSpec.params
                .filter((p) => p.name !== 'p' && paramVisible(p, stage.params))
                .map((p) => (
                  <ParamControl
                    key={p.name}
                    spec={p}
                    value={stage.params[p.name]}
                    onChange={(v) =>
                      setStages((x) =>
                        x.map((y) =>
                          y.rowId === stage.rowId
                            ? { ...y, params: { ...y.params, [p.name]: v } }
                            : y,
                        ),
                      )
                    }
                  />
                ))}
            </div>
          )}
        </div>

        {/* --------------------------------------------- batch generate */}
        <div className="shrink-0 space-y-2 border-t border-border p-2">
          <div className="flex items-center gap-1">
            <Package className="h-3.5 w-3.5 text-primary" />
            <span className="text-xs font-semibold">Generate dataset</span>
          </div>

          {images.length > 1 && (
            <div className="space-y-1">
              <Label>Source images ({selectedImages.length})</Label>
              <div className="flex flex-wrap gap-1">
                {images.map((im) => (
                  <button
                    key={im.image_id}
                    onClick={() =>
                      setSelectedImages((s) =>
                        s.includes(im.image_id)
                          ? s.filter((x) => x !== im.image_id)
                          : [...s, im.image_id],
                      )
                    }
                    className={cn(
                      'h-7 w-9 overflow-hidden rounded border transition-colors',
                      selectedImages.includes(im.image_id)
                        ? 'border-primary'
                        : 'border-border opacity-50',
                    )}
                    title={im.name}
                  >
                    {im.thumbnail && (
                      <img src={im.thumbnail.url} alt="" className="h-full w-full object-cover" />
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <Label>Variants per image</Label>
              <span className="text-[10px] tabular text-muted-foreground">{variantsPerImage}</span>
            </div>
            <Slider
              value={[variantsPerImage]}
              min={1}
              max={50}
              step={1}
              onValueChange={([v]) => setVariantsPerImage(v)}
              aria-label="Variants per image"
            />
          </div>

          <Button
            size="sm"
            className="w-full"
            onClick={generate}
            disabled={!stages.length || !selectedImages.length || gen !== null}
          >
            {gen ? <Spinner /> : <Download className="h-3.5 w-3.5" />}
            {gen
              ? `${Math.round(gen.progress * 100)}% ${gen.message}`
              : `Generate ${selectedImages.length * variantsPerImage} images`}
          </Button>

          <p className="text-[9px] leading-snug text-muted-foreground">
            The zip includes a manifest.json recording the exact sampled
            parameters for every output, so any image can be reproduced.
          </p>
        </div>
      </aside>

      {/* -------------------------------------------------------- gallery */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-9 shrink-0 items-center gap-2 border-b border-border bg-card px-2">
          <span className="text-xs font-semibold">Variant gallery</span>
          <div className="flex-1" />
          <div className="flex items-center gap-1">
            <Label>count</Label>
            <Input
              type="number"
              min={1}
              max={48}
              value={count}
              onChange={(e) => setCount(Math.max(1, Math.min(48, Number(e.target.value))))}
              className="h-6 w-14 text-[11px]"
              aria-label="Variant count"
            />
          </div>
          <div className="flex items-center gap-1">
            <Label>seed</Label>
            <Input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="h-6 w-20 text-[11px]"
              aria-label="Seed"
            />
            <Tooltip content="New seed">
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={() => setSeed(Math.floor(Math.random() * 1e6))}
                aria-label="Randomise seed"
              >
                <Dices className="h-3.5 w-3.5" />
              </Button>
            </Tooltip>
          </div>
          <Tooltip content="Resample">
            <Button variant="ghost" size="icon-sm" onClick={sample} disabled={busy} aria-label="Resample">
              <RefreshCw className={cn('h-3.5 w-3.5', busy && 'animate-spin')} />
            </Button>
          </Tooltip>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-3 scrollbar-thin">
          {error && (
            <div className="mb-3 rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {error}
            </div>
          )}

          {!stages.length ? (
            <EmptyState icon={<Dices className="h-8 w-8" />} title="Build a recipe to see variants">
              Add stages on the left. Every variant is produced by sampling the
              recipe with a different sub-seed, and the same seed always
              reproduces the same set.
            </EmptyState>
          ) : (
            <>
              <div className="mb-3 flex items-center gap-2">
                <img
                  src={active.thumbnail?.url}
                  alt="Original"
                  className="h-16 w-20 rounded border border-primary object-cover"
                />
                <div>
                  <div className="text-[10px] font-semibold">Original</div>
                  <div className="text-[10px] text-muted-foreground">{active.name}</div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                {variants.map((v) => (
                  <div
                    key={v.index}
                    className={cn(
                      'group relative overflow-hidden rounded border transition-colors',
                      pinned.includes(v.index) ? 'border-primary' : 'border-border',
                    )}
                  >
                    <img src={v.render.url} alt={`Variant ${v.index}`} className="aspect-[4/3] w-full object-cover" />
                    <div className="absolute inset-x-0 bottom-0 flex items-center gap-0.5 bg-background/85 px-1 py-0.5 opacity-0 backdrop-blur transition-opacity group-hover:opacity-100">
                      <span className="text-[9px] tabular text-muted-foreground">#{v.index}</span>
                      <div className="flex-1" />
                      <Tooltip content="Pin this variant">
                        <Button
                          variant="ghost"
                          size="icon-sm"
                          onClick={() =>
                            setPinned((p) =>
                              p.includes(v.index) ? p.filter((x) => x !== v.index) : [...p, v.index],
                            )
                          }
                          aria-label="Pin variant"
                        >
                          <Pin className={cn('h-3 w-3', pinned.includes(v.index) && 'text-primary')} />
                        </Button>
                      </Tooltip>
                      <Tooltip content="Load this exact recipe into the Studio pipeline">
                        <Button variant="ghost" size="icon-sm" onClick={() => promote(v)} aria-label="Promote to Studio">
                          <Plus className="h-3 w-3" />
                        </Button>
                      </Tooltip>
                    </div>
                    <AppliedBadges applied={v.applied} />
                  </div>
                ))}
                {busy && !variants.length && (
                  <div className="col-span-full flex justify-center py-8">
                    <Spinner className="h-5 w-5" />
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function AppliedBadges({ applied }: { applied: Record<string, Record<string, unknown>> }) {
  // Which stages actually fired, and with what sampled values. This is what
  // makes a variant reproducible rather than mysterious.
  const entries = Object.entries(applied ?? {}).filter(
    ([, v]) => (v as { applied?: boolean }).applied !== false,
  )
  if (!entries.length) return null
  return (
    <Tooltip
      content={
        <div className="space-y-0.5">
          {entries.map(([k, v]) => (
            <div key={k} className="text-[10px]">
              <span className="font-medium">{k.split('-').slice(2).join('-') || k}</span>{' '}
              {Object.entries(v)
                .filter(([pk]) => pk !== 'applied')
                .map(([pk, pv]) => `${pk}=${typeof pv === 'number' ? fmtNum(pv, 3) : String(pv)}`)
                .join(' ')}
            </div>
          ))}
        </div>
      }
    >
      <div className="absolute right-1 top-1 cursor-help rounded bg-background/80 px-1 text-[9px] tabular backdrop-blur">
        {entries.length}
      </div>
    </Tooltip>
  )
}
