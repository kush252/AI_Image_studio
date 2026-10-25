import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'
import type { ParamSpec, StackOp } from './types'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

let uidCounter = 0
export function uid(prefix = 'op'): string {
  uidCounter += 1
  return `${prefix}-${Date.now().toString(36)}-${uidCounter.toString(36)}`
}

export function defaultsFor(params: ParamSpec[]): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const p of params) out[p.name] = p.default
  return out
}

export function makeOp(
  opId: string,
  params: ParamSpec[],
  overrides?: Partial<StackOp>,
): StackOp {
  return {
    uid: uid(),
    op_id: opId,
    params: { ...defaultsFor(params), ...(overrides?.params ?? {}) },
    enabled: true,
    opacity: 1,
    blend: 'normal',
    mask: null,
    collapsed: false,
    ...overrides,
  }
}

/** Show a param only when its `depends_on` condition is met. */
export function paramVisible(p: ParamSpec, values: Record<string, unknown>): boolean {
  if (!p.depends_on) return true
  const actual = values[p.depends_on.param]
  const want = p.depends_on.equals
  // Enum values arrive as strings; booleans must compare loosely so that
  // depends_on {equals: false} matches an unchecked switch.
  if (typeof want === 'boolean') return Boolean(actual) === want
  return String(actual) === String(want)
}

export function fmtNum(v: unknown, digits = 4): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '--'
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1000) return v.toFixed(0)
  if (a >= 1) return v.toFixed(Math.min(digits, 2))
  if (a < 1e-4) return v.toExponential(2)
  return v.toFixed(digits)
}

export function fmtPct(v: unknown, digits = 2): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '--'
  return `${(v * 100).toFixed(digits)}%`
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

export function fmtMs(n: number): string {
  if (n < 1) return '<1 ms'
  if (n < 1000) return `${n.toFixed(0)} ms`
  return `${(n / 1000).toFixed(2)} s`
}

export const SEVERITY_ORDER = ['severe', 'moderate', 'minor', 'info', 'none'] as const

export function severityClass(sev: string): string {
  switch (sev) {
    case 'severe': return 'text-sev-severe'
    case 'moderate': return 'text-sev-moderate'
    case 'minor': return 'text-sev-minor'
    case 'info': return 'text-sev-info'
    default: return 'text-sev-none'
  }
}

export function severityBg(sev: string): string {
  switch (sev) {
    case 'severe': return 'bg-sev-severe/15 border-sev-severe/40'
    case 'moderate': return 'bg-sev-moderate/15 border-sev-moderate/40'
    case 'minor': return 'bg-sev-minor/15 border-sev-minor/40'
    case 'info': return 'bg-sev-info/15 border-sev-info/40'
    default: return 'bg-sev-none/15 border-sev-none/40'
  }
}

export function scoreColor(score: number): string {
  if (score >= 85) return 'hsl(var(--sev-none))'
  if (score >= 70) return 'hsl(var(--sev-info))'
  if (score >= 55) return 'hsl(var(--sev-minor))'
  if (score >= 35) return 'hsl(var(--sev-moderate))'
  return 'hsl(var(--sev-severe))'
}

export function debounce<T extends (...args: never[]) => void>(fn: T, ms: number) {
  let t: ReturnType<typeof setTimeout> | undefined
  const wrapped = (...args: Parameters<T>) => {
    if (t) clearTimeout(t)
    t = setTimeout(() => fn(...args), ms)
  }
  wrapped.cancel = () => { if (t) clearTimeout(t) }
  return wrapped as T & { cancel: () => void }
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(Math.max(v, lo), hi)
}

/** Snap to the step grid a ParamSpec declares, keeping odd-int params odd. */
export function snapToSpec(p: ParamSpec, raw: number): number {
  const lo = p.min ?? 0
  const hi = p.max ?? 1
  let v = clamp(raw, lo, hi)
  if (p.kind === 'odd_int') {
    v = Math.round(v)
    if (v % 2 === 0) v += 1
    return clamp(v, Math.max(lo, 1), hi)
  }
  if (p.kind === 'int' || p.kind === 'seed') return Math.round(v)
  const step = p.step ?? 0.01
  if (step > 0) v = Math.round(v / step) * step
  // Trim binary float noise introduced by the division above.
  const dec = Math.max(0, Math.ceil(-Math.log10(step)) + 1)
  return Number(clamp(v, lo, hi).toFixed(dec))
}

export function titleCase(s: string): string {
  return s.replace(/[_-]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

export function downloadText(filename: string, text: string, mime = 'text/plain') {
  const blob = new Blob([text], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

/** Strip client-only fields from an op stack before sending it to the server. */
export function wireOpsSafe(ops: StackOp[]) {
  return ops.map((o) => ({
    op_id: o.op_id,
    params: o.params,
    enabled: o.enabled,
    opacity: o.opacity,
    blend: o.blend,
    mask: o.mask,
    uid: o.uid,
  }))
}

export function downloadUrl(filename: string, url: string) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
}
