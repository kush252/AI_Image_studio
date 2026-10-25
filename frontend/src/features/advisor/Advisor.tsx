/** Advisor: measure -> diagnose -> derive -> apply -> verify.
 *  Implements the UI described in docs/advisor-spec.md and PLAN.md Phase 8.
 *
 *  Every finding shows the number that triggered it, the expected range, and
 *  the derivation of the parameters it proposes. That is what makes the
 *  section auditable rather than a list of plausible-sounding suggestions.
 */
import {
  AlertTriangle, Check, CheckCheck, ChevronDown, Eye, Info, RefreshCw,
  RotateCcw, ShieldAlert, Sparkles, TrendingDown, TrendingUp, X,
} from 'lucide-react'
import * as React from 'react'

import { ImageCanvas } from '@/components/canvas/ImageCanvas'
import {
  DimensionBars, DimensionRadar, HealthRing, MetricGauge,
} from '@/components/charts/ScoreCharts'
import { Formula } from '@/components/controls/OperatorPanel'
import { Dropzone } from '@/components/layout/Shell'
import {
  Badge, Button, EmptyState, Separator, Spinner, Tooltip,
} from '@/components/ui'
import { api, wireOps } from '@/lib/api'
import type { Finding, StackOp, Verification } from '@/lib/types'
import { cn, fmtNum, severityBg, severityClass, titleCase, uid } from '@/lib/utils'
import { useCatalogue, usePipeline, useSession, useUI } from '@/store'

const STRENGTHS = [
  { id: 'subtle', label: 'Subtle', value: 0.5, help: 'Half the derived amount.' },
  { id: 'balanced', label: 'Balanced', value: 1.0, help: 'The derived amount as measured.' },
  { id: 'strong', label: 'Strong', value: 1.6, help: '1.6x the derived amount.' },
]

export function Advisor() {
  const active = useSession((s) => s.images.find((i) => i.image_id === s.activeId) ?? null)
  if (!active) return <Dropzone />
  return (
    <div className="flex min-h-0 flex-1">
      <div className="min-w-0 flex-1">
        <ImageCanvas />
      </div>
      <AdvisorPanel />
    </div>
  )
}

function AdvisorPanel() {
  const activeId = useSession((s) => s.activeId)
  const {
    diagnosis, setDiagnosis, diagnosing, setDiagnosing, verification,
    setVerification, ops, setOps, commit, dismissed, dismissFinding,
    resetDismissed, clear,
  } = usePipeline()
  const setShowOverlay = useUI((s) => s.setShowOverlay)

  const [strength, setStrength] = React.useState(1.0)
  const [includeOptIn, setIncludeOptIn] = React.useState(false)
  const [showLowConfidence, setShowLowConfidence] = React.useState(false)
  const [verifying, setVerifying] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [appliedIds, setAppliedIds] = React.useState<string[]>([])

  const runDiagnose = React.useCallback(async () => {
    if (!activeId) return
    setDiagnosing(true)
    setError(null)
    try {
      const d = await api.diagnose({
        image_id: activeId,
        strength,
        include_opt_in: includeOptIn,
        include_maps: true,
      })
      setDiagnosis(d)
      setVerification(null)
      setAppliedIds([])
      resetDismissed()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setDiagnosing(false)
    }
  }, [activeId, strength, includeOptIn, setDiagnosing, setDiagnosis, setVerification, resetDismissed])

  // Diagnose on image change, since that is what the user is asking for by
  // switching to this section.
  React.useEffect(() => {
    if (activeId && !diagnosis) void runDiagnose()
  }, [activeId]) // eslint-disable-line react-hooks/exhaustive-deps

  const runVerify = React.useCallback(async () => {
    if (!activeId) return
    const current = usePipeline.getState().ops
    if (!current.length) {
      setVerification(null)
      return
    }
    setVerifying(true)
    try {
      const v = await api.verify({
        image_id: activeId,
        ops: wireOps(current),
        attribute_regressions: true,
      })
      setVerification(v)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setVerifying(false)
    }
  }, [activeId, setVerification])

  const applyFix = (f: Finding) => {
    if (!f.fix) return
    commit()
    const added: StackOp[] = f.fix.ops.map((o) => ({
      uid: uid('fix'),
      op_id: o.op_id,
      params: { ...o.params },
      enabled: true,
      opacity: 1,
      blend: 'normal',
      mask: (o.mask as Record<string, unknown>) ?? null,
      from_finding: f.id,
    }))
    setOps([...usePipeline.getState().ops, ...added], false)
    setAppliedIds((a) => [...a, f.id])
  }

  const applyAll = () => {
    if (!diagnosis?.recipe.length) return
    commit()
    const added: StackOp[] = diagnosis.recipe.map((o) => ({
      uid: uid('fix'),
      op_id: o.op_id,
      params: { ...o.params },
      enabled: true,
      opacity: 1,
      blend: 'normal',
      mask: (o.mask as Record<string, unknown>) ?? null,
      from_finding: o.from_finding,
    }))
    setOps(added, false)
    setAppliedIds(diagnosis.findings.filter((f) => f.fix).map((f) => f.id))
  }

  if (!activeId) return null

  const findings = (diagnosis?.findings ?? []).filter((f) => !dismissed.includes(f.id))
  const confident = findings.filter((f) => f.confidence >= 0.45)
  const lowConf = findings.filter((f) => f.confidence < 0.45)
  const actionable = confident.filter((f) => f.fix)
  const warnings = confident.filter((f) => !f.fix)

  return (
    <aside className="flex w-[388px] shrink-0 flex-col border-l border-border bg-card">
      {/* ------------------------------------------------------- header */}
      <div className="flex h-9 shrink-0 items-center gap-1 border-b border-border px-2">
        <Sparkles className="h-3.5 w-3.5 text-primary" />
        <span className="text-xs font-semibold">Advisor</span>
        <div className="flex-1" />
        <Tooltip content="Re-measure this image">
          <Button variant="ghost" size="icon-sm" onClick={runDiagnose} disabled={diagnosing} aria-label="Re-analyse">
            <RefreshCw className={cn('h-3.5 w-3.5', diagnosing && 'animate-spin')} />
          </Button>
        </Tooltip>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto scrollbar-thin">
        {diagnosing && !diagnosis && (
          <div className="flex flex-col items-center gap-2 py-12">
            <Spinner className="h-6 w-6" />
            <span className="text-xs text-muted-foreground">Measuring the image...</span>
          </div>
        )}

        {error && (
          <div className="m-2 rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {error}
          </div>
        )}

        {diagnosis && (
          <>
            {/* ------------------------------------------- score card */}
            <div className="space-y-3 border-b border-border p-3">
              <div className="flex items-center gap-4">
                <HealthRing score={diagnosis.scores.overall} />
                <div className="min-w-0 flex-1 space-y-1">
                  <p className="text-[11px] leading-snug text-muted-foreground">
                    {confident.length === 0
                      ? 'Every measured dimension is inside its expected range.'
                      : `${confident.length} finding${confident.length === 1 ? '' : 's'}: ` +
                        `${actionable.length} with a fix, ${warnings.length} advisory.`}
                  </p>
                  {diagnosis.scores.worst_dimension && confident.length > 0 && (
                    <p className="text-[10px] text-muted-foreground">
                      Weakest dimension:{' '}
                      <span className={severityClass(
                        diagnosis.scores.dimensions[diagnosis.scores.worst_dimension]?.severity ?? 'none',
                      )}>
                        {titleCase(diagnosis.scores.worst_dimension)}
                      </span>
                    </p>
                  )}
                  {diagnosis.document_mode && (
                    <Badge variant="outline">document mode</Badge>
                  )}
                  {diagnosis.monochrome && <Badge variant="outline">monochrome</Badge>}
                </div>
              </div>

              <DimensionRadar scores={diagnosis.scores} verification={verification} height={190} />
              <DimensionBars scores={diagnosis.scores} verification={verification} />
            </div>

            {/* --------------------------------------------- controls */}
            <div className="space-y-2 border-b border-border p-3">
              <div className="flex items-center gap-1">
                <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                  Strength
                </span>
                <div className="flex-1" />
                <div className="flex gap-0.5">
                  {STRENGTHS.map((s) => (
                    <Tooltip key={s.id} content={s.help}>
                      <button
                        onClick={() => setStrength(s.value)}
                        className={cn(
                          'rounded px-1.5 py-0.5 text-[10px] transition-colors',
                          strength === s.value
                            ? 'bg-primary/20 text-primary'
                            : 'text-muted-foreground hover:bg-accent',
                        )}
                      >
                        {s.label}
                      </button>
                    </Tooltip>
                  ))}
                </div>
              </div>

              <label className="flex cursor-pointer items-center gap-1.5 text-[10px] text-muted-foreground">
                <input
                  type="checkbox"
                  checked={includeOptIn}
                  onChange={(e) => setIncludeOptIn(e.target.checked)}
                  className="h-3 w-3 accent-[hsl(var(--primary))]"
                />
                Include geometric fixes (these crop and resample, so they are a
                compositional choice)
              </label>

              <div className="flex gap-1.5">
                <Button
                  size="sm"
                  className="flex-1"
                  onClick={applyAll}
                  disabled={!diagnosis.recipe.length}
                >
                  <CheckCheck className="h-3.5 w-3.5" />
                  Apply all ({diagnosis.recipe.length})
                </Button>
                <Tooltip content="Re-measure the processed image and compare">
                  <Button size="sm" variant="outline" onClick={runVerify} disabled={verifying || !ops.length}>
                    {verifying ? <Spinner /> : <Check className="h-3.5 w-3.5" />}
                    Verify
                  </Button>
                </Tooltip>
                <Tooltip content="Clear the pipeline">
                  <Button size="icon-sm" variant="ghost" onClick={clear} disabled={!ops.length} aria-label="Clear pipeline">
                    <RotateCcw className="h-3.5 w-3.5" />
                  </Button>
                </Tooltip>
              </div>

              {diagnosis.recipe.length > 0 && (
                <RecipePreview recipe={diagnosis.recipe} />
              )}
            </div>

            {/* ------------------------------------------ verification */}
            {verification && <VerificationPanel />}

            {/* ---------------------------------------------- findings */}
            <div className="space-y-2 p-3">
              {confident.length === 0 && !diagnosing && (
                <EmptyState
                  icon={<Check className="h-7 w-7 text-sev-none" />}
                  title="Nothing to fix"
                  className="py-6"
                >
                  No measured dimension crossed its threshold. A diagnostic tool
                  that always finds something wrong is not diagnosing.
                </EmptyState>
              )}

              {actionable.map((f) => (
                <FindingCard
                  key={f.id}
                  finding={f}
                  applied={appliedIds.includes(f.id)}
                  maps={diagnosis.maps}
                  onApply={() => applyFix(f)}
                  onDismiss={() => dismissFinding(f.id)}
                  onShowMap={setShowOverlay}
                />
              ))}

              {warnings.length > 0 && (
                <>
                  <div className="flex items-center gap-2 pt-1">
                    <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                      Advisory
                    </span>
                    <Separator className="flex-1" />
                  </div>
                  {warnings.map((f) => (
                    <FindingCard
                      key={f.id}
                      finding={f}
                      applied={false}
                      maps={diagnosis.maps}
                      onDismiss={() => dismissFinding(f.id)}
                      onShowMap={setShowOverlay}
                    />
                  ))}
                </>
              )}

              {lowConf.length > 0 && (
                <div className="space-y-2">
                  <button
                    onClick={() => setShowLowConfidence((v) => !v)}
                    className="flex w-full items-center gap-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground"
                  >
                    <ChevronDown className={cn('h-3 w-3 transition-transform', !showLowConfidence && '-rotate-90')} />
                    Low confidence ({lowConf.length})
                  </button>
                  {showLowConfidence &&
                    lowConf.map((f) => (
                      <FindingCard
                        key={f.id}
                        finding={f}
                        applied={appliedIds.includes(f.id)}
                        maps={diagnosis.maps}
                        onApply={f.fix ? () => applyFix(f) : undefined}
                        onDismiss={() => dismissFinding(f.id)}
                        onShowMap={setShowOverlay}
                      />
                    ))}
                </div>
              )}

              {diagnosis.notes.map((n, i) => (
                <div
                  key={i}
                  className="flex gap-1.5 rounded border border-border/60 bg-muted/30 p-2 text-[10px] leading-snug text-muted-foreground"
                >
                  <Info className="mt-px h-3 w-3 shrink-0" />
                  {n}
                </div>
              ))}
            </div>

            <div className="border-t border-border p-3">
              <p className="text-[9px] leading-relaxed text-muted-foreground">
                Every finding above cites the metric that triggered it and the
                range it was compared against; every parameter is derived by a
                documented formula. No learned models are involved — all of it
                comes from statistics measured on this image.
              </p>
            </div>
          </>
        )}
      </div>
    </aside>
  )
}

function RecipePreview({ recipe }: { recipe: (StackOp & { stage: number })[] }) {
  const byId = useCatalogue((s) => s.byId)
  const [open, setOpen] = React.useState(false)
  return (
    <div className="rounded border border-border/60 bg-muted/30">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1 px-2 py-1.5 text-left"
      >
        <ChevronDown className={cn('h-3 w-3 shrink-0 transition-transform', !open && '-rotate-90')} />
        <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          Proposed recipe
        </span>
        <div className="flex-1" />
        <span className="text-[9px] text-muted-foreground">ordered</span>
      </button>
      {open && (
        <div className="space-y-1 px-2 pb-2">
          <p className="text-[9px] leading-snug text-muted-foreground">
            Canonical order: geometry, white balance, denoise, tone, local tone,
            contrast, colour, sharpen. Denoising runs before any contrast boost
            because contrast amplifies noise; sharpening runs last because it
            amplifies whatever precedes it.
          </p>
          {recipe.map((o, i) => (
            <div key={o.uid ?? i} className="flex items-baseline gap-1.5 text-[10px]">
              <span className="w-3 shrink-0 tabular text-muted-foreground">{i + 1}</span>
              <span className="font-medium">{byId[o.op_id]?.name ?? o.op_id}</span>
              <span className="truncate text-muted-foreground">
                {Object.entries(o.params)
                  .filter(([, v]) => typeof v === 'number' || typeof v === 'string')
                  .slice(0, 3)
                  .map(([k, v]) => `${k}=${typeof v === 'number' ? fmtNum(v, 3) : v}`)
                  .join(' ')}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function FindingCard({
  finding, applied, maps, onApply, onDismiss, onShowMap,
}: {
  finding: Finding
  applied: boolean
  maps: Record<string, { url: string }>
  onApply?: () => void
  onDismiss: () => void
  onShowMap: (name: string | null) => void
}) {
  const [open, setOpen] = React.useState(false)
  const byId = useCatalogue((s) => s.byId)
  const ev = finding.evidence

  // Pick the evidence map that matches this finding's dimension.
  const mapName = React.useMemo(() => {
    const d = finding.dimension
    if (finding.id.startsWith('NSE_PERIODIC')) return 'spectrum'
    if (d === 'noise') return 'noise'
    if (d === 'uniformity') return finding.id === 'UNI_HAZE' ? 'dark_channel' : undefined
    if (d === 'exposure') return 'clipping'
    if (d === 'sharpness') return 'saliency'
    return undefined
  }, [finding])

  return (
    <div className={cn('rounded border', severityBg(finding.severity))}>
      <div className="flex items-start gap-1.5 p-2">
        <div className={cn('mt-0.5 shrink-0', severityClass(finding.severity))}>
          {finding.warning_only ? <ShieldAlert className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="truncate text-xs font-semibold">{finding.title}</span>
            <Badge variant="outline">{finding.severity}</Badge>
            {finding.confidence < 0.99 && (
              <Tooltip content="Confidence that this finding is real, from how unambiguous the evidence is.">
                <span className="cursor-help text-[9px] tabular text-muted-foreground">
                  {Math.round(finding.confidence * 100)}%
                </span>
              </Tooltip>
            )}
          </div>
          <p className="mt-0.5 text-[10px] leading-relaxed text-foreground/80">
            {finding.explanation}
          </p>
        </div>
        <Tooltip content="Dismiss">
          <Button variant="ghost" size="icon-sm" onClick={onDismiss} aria-label="Dismiss finding">
            <X className="h-3 w-3" />
          </Button>
        </Tooltip>
      </div>

      {/* Evidence: the number, and where it sits against the expected band. */}
      {typeof ev.value === 'number' && ev.expected && (
        <div className="px-2 pb-1.5">
          <MetricGauge
            label={ev.metric?.replace(/_/g, ' ') ?? 'metric'}
            value={ev.value}
            expected={ev.expected}
            help={`Measured ${fmtNum(ev.value)}; expected ${fmtNum(ev.expected[0])} to ${fmtNum(ev.expected[1])}.`}
          />
        </div>
      )}

      {ev.secondary && Object.keys(ev.secondary).length > 0 && (
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 px-2 pb-1.5">
          {Object.entries(ev.secondary).map(([k, v]) => (
            <span key={k} className="text-[9px] tabular text-muted-foreground">
              {k.replace(/_/g, ' ')} {fmtNum(v, 4)}
            </span>
          ))}
        </div>
      )}

      {ev.peaks && ev.peaks.length > 0 && (
        <div className="px-2 pb-1.5">
          <span className="text-[9px] text-muted-foreground">
            Peaks at{' '}
            {ev.peaks.slice(0, 3).map((p) => `(${p.fx.toFixed(3)}, ${p.fy.toFixed(3)})`).join(', ')}{' '}
            cycles/px
          </span>
        </div>
      )}

      <div className="flex items-center gap-1 border-t border-border/40 px-2 py-1.5">
        {onApply && (
          <Button
            size="xs"
            variant={applied ? 'secondary' : 'default'}
            onClick={onApply}
            disabled={applied}
          >
            {applied ? <Check className="h-3 w-3" /> : <Sparkles className="h-3 w-3" />}
            {applied ? 'Applied' : (finding.fix?.label ?? 'Apply')}
          </Button>
        )}
        {mapName && maps[mapName] && (
          <Tooltip content="Show the evidence for this on the canvas">
            <Button size="xs" variant="ghost" onClick={() => onShowMap(mapName)}>
              <Eye className="h-3 w-3" />
              Evidence
            </Button>
          </Tooltip>
        )}
        <div className="flex-1" />
        {finding.fix && (
          <Button size="xs" variant="ghost" onClick={() => setOpen((v) => !v)}>
            <ChevronDown className={cn('h-3 w-3 transition-transform', !open && '-rotate-90')} />
            How
          </Button>
        )}
      </div>

      {open && finding.fix && (
        <div className="space-y-2 border-t border-border/40 bg-background/40 p-2">
          <div>
            <span className="text-[9px] font-semibold uppercase tracking-wide text-muted-foreground">
              Operators
            </span>
            {finding.fix.ops.map((o, i) => {
              const spec = byId[o.op_id]
              return (
                <div key={i} className="mt-1 space-y-0.5">
                  <div className="flex items-center gap-1">
                    <span className="text-[10px] font-medium">{spec?.name ?? o.op_id}</span>
                    {spec?.syllabus && <Badge variant="syllabus">S</Badge>}
                  </div>
                  <div className="flex flex-wrap gap-x-2 text-[9px] tabular text-muted-foreground">
                    {Object.entries(o.params).map(([k, v]) => (
                      <span key={k}>
                        {k}={typeof v === 'number' ? fmtNum(v, 4) : String(v)}
                      </span>
                    ))}
                  </div>
                  {spec?.formula_latex && <Formula latex={spec.formula_latex} display={false} />}
                </div>
              )
            })}
          </div>
          <div>
            <span className="text-[9px] font-semibold uppercase tracking-wide text-muted-foreground">
              Derivation
            </span>
            <p className="mt-0.5 text-[10px] leading-relaxed text-foreground/75">
              {finding.fix.derivation}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

function VerificationPanel() {
  const { verification, ops, setOps, commit } = usePipeline()
  const byId = useCatalogue((s) => s.byId)
  if (!verification) return null
  const v = verification
  const up = v.overall_delta > 0

  const revert = (uidToRemove?: string) => {
    commit()
    setOps(uidToRemove ? ops.filter((o) => o.uid !== uidToRemove) : [], false)
  }

  return (
    <div className="space-y-2 border-b border-border bg-muted/20 p-3">
      <div className="flex items-center gap-2">
        {up ? (
          <TrendingUp className="h-4 w-4 text-sev-none" />
        ) : (
          <TrendingDown className="h-4 w-4 text-sev-moderate" />
        )}
        <span className="text-xs font-semibold">Verification</span>
        <div className="flex-1" />
        <span className={cn('text-sm font-semibold tabular', up ? 'text-sev-none' : 'text-sev-moderate')}>
          {v.overall_before.toFixed(1)} → {v.overall_after.toFixed(1)}
        </span>
      </div>

      <p className="text-[10px] leading-relaxed text-muted-foreground">{v.summary}</p>

      {v.regressions.length > 0 && (
        <div className="space-y-1 rounded border border-sev-moderate/40 bg-sev-moderate/10 p-2">
          <div className="flex items-center gap-1">
            <AlertTriangle className="h-3 w-3 text-sev-moderate" />
            <span className="text-[10px] font-semibold text-sev-moderate">Regressions</span>
          </div>
          {v.regressions.map((r) => (
            <div key={r.dimension} className="space-y-0.5">
              <div className="flex items-baseline gap-1.5 text-[10px]">
                <span className="font-medium">{titleCase(r.dimension)}</span>
                <span className="tabular text-sev-moderate">{r.delta.toFixed(1)}</span>
                <span className="tabular text-muted-foreground">
                  ({r.before.toFixed(1)} → {r.after.toFixed(1)})
                </span>
              </div>
              {r.likely_cause && (
                <div className="flex items-center gap-1 pl-1">
                  <span className="text-[9px] text-muted-foreground">
                    likely {byId[r.likely_cause.op_id]?.name ?? r.likely_cause.op_id}; removing it
                    recovers {r.likely_cause.recovered.toFixed(1)}
                  </span>
                  <Button size="xs" variant="ghost" onClick={() => revert(r.likely_cause!.uid)}>
                    Remove
                  </Button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {v.tradeoffs.length > 0 && (
        <div className="space-y-0.5 rounded border border-border/60 p-2">
          <Tooltip content="Inherent to the operators used, not faults. Denoising necessarily removes high-frequency detail; sharpening necessarily amplifies noise.">
            <span className="cursor-help text-[10px] font-semibold text-muted-foreground">
              Expected trade-offs
            </span>
          </Tooltip>
          {v.tradeoffs.map((t) => (
            <div key={t.dimension} className="flex items-baseline gap-1.5 text-[10px] text-muted-foreground">
              <span>{titleCase(t.dimension)}</span>
              <span className="tabular">{t.delta.toFixed(1)}</span>
            </div>
          ))}
        </div>
      )}

      {v.metrics_moved.length > 0 && <MetricsMoved moved={v.metrics_moved} />}
    </div>
  )
}

type MovedMetric = Verification['metrics_moved'][number]

function MetricsMoved({ moved }: { moved: MovedMetric[] }) {
  const [open, setOpen] = React.useState(false)
  const entered = moved.filter((m) => m.entered_range).length
  const left = moved.filter((m) => m.left_range).length
  return (
    <div className="rounded border border-border/60">
      <button onClick={() => setOpen((v) => !v)} className="flex w-full items-center gap-1 px-2 py-1.5">
        <ChevronDown className={cn('h-3 w-3 transition-transform', !open && '-rotate-90')} />
        <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          Metrics moved ({moved.length})
        </span>
        <div className="flex-1" />
        {entered > 0 && <span className="text-[9px] text-sev-none">{entered} fixed</span>}
        {left > 0 && <span className="text-[9px] text-sev-moderate">{left} broke</span>}
      </button>
      {open && (
        <div className="max-h-56 space-y-0.5 overflow-y-auto px-2 pb-2 scrollbar-thin">
          {moved.map((m) => (
            <div key={m.metric} className="flex items-baseline gap-1.5 text-[9px] tabular">
              <span
                className={cn(
                  'w-3 shrink-0',
                  m.entered_range ? 'text-sev-none' : m.left_range ? 'text-sev-moderate' : 'text-transparent',
                )}
              >
                {m.entered_range ? '✓' : m.left_range ? '!' : '·'}
              </span>
              <span className="min-w-0 flex-1 truncate text-muted-foreground">
                {m.metric.replace(/_/g, ' ')}
              </span>
              <span>{fmtNum(m.before, 3)}</span>
              <span className="text-muted-foreground">→</span>
              <span>{fmtNum(m.after, 3)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
