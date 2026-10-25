// Mirrors the backend schemas. PLAN.md Sections 4 and 5.

export type ParamKind =
  | 'float' | 'int' | 'odd_int' | 'bool' | 'enum' | 'color' | 'point'
  | 'points' | 'rect' | 'line' | 'range' | 'matrix' | 'curve' | 'image_ref'
  | 'mask_ref' | 'freq_mask' | 'lut_file' | 'seed' | 'string' | 'any'

export interface EnumOption { value: string; label: string; help?: string; icon?: string }

export interface ParamSpec {
  name: string
  label: string
  kind: ParamKind
  default: unknown
  help: string
  group: string
  advanced: boolean
  scale_aware: boolean
  auto: boolean
  unit: string
  depends_on?: { param: string; equals: unknown }
  min?: number
  max?: number
  step?: number
  log_scale?: boolean
  options?: EnumOption[]
  rows?: number
  cols?: number
  presets?: { label: string; value: number[][] }[]
  channels?: string[]
  min_count?: number
  max_count?: number
}

export interface OperationSpec {
  id: string
  name: string
  category: string
  params: ParamSpec[]
  summary: string
  formula_latex: string
  when_to_use: string
  references: string[]
  domain: 'srgb' | 'linear' | 'either'
  is_view: boolean
  syllabus: boolean
  requires: { channels?: number; binary?: boolean; second_image?: boolean; second_image_param?: string; strict?: boolean }
  tags: string[]
  cost: 'fast' | 'medium' | 'slow'
  has_auto: boolean
}

export interface CategoryInfo {
  id: string
  label: string
  order: number
  icon: string
  blurb: string
  count: number
}

export interface Catalogue {
  categories: CategoryInfo[]
  operations: OperationSpec[]
  total: number
  syllabus_count: number
  blend_modes: { value: string; label: string }[]
}

export interface StackOp {
  uid: string
  op_id: string
  params: Record<string, unknown>
  enabled: boolean
  opacity: number
  blend: string
  mask: Record<string, unknown> | null
  collapsed?: boolean
  name?: string
  from_finding?: string
}

export interface RenderRef {
  render_id: string
  url: string
  width: number
  height: number
  bytes: number
}

export interface ImageInfo {
  image_id: string
  name: string
  width: number
  height: number
  channels: number
  megapixels: number
  proxy_width: number
  proxy_height: number
  proxy_scale: number
  source_format: string
  source_bit_depth: number
  had_alpha: boolean
  was_grayscale: boolean
  exif: Record<string, unknown>
  created_at: number
  preview?: RenderRef
  thumbnail?: RenderRef
}

export interface PreviewStats {
  mean: number
  std: number
  min: number
  max: number
  clip_low: number
  clip_high: number
  histogram: number[]
  histogram_r?: number[]
  histogram_g?: number[]
  histogram_b?: number[]
}

export interface PreviewResponse {
  seq: number
  render: RenderRef
  timings: { op_id: string; ms: number; cached: boolean }[]
  total_ms: number
  cache_hits: number
  data: Record<string, Record<string, unknown>>
  overlays: Record<string, RenderRef>
  notes: string[]
  stats?: PreviewStats
  scale_factor: number
  width: number
  height: number
}

export type Severity = 'none' | 'info' | 'minor' | 'moderate' | 'severe'

export interface Finding {
  id: string
  dimension: string
  severity: Severity
  confidence: number
  title: string
  explanation: string
  evidence: {
    metric?: string
    value?: number
    expected?: [number, number]
    secondary?: Record<string, number>
    peaks?: { x: number; y: number; fx: number; fy: number; prominence: number }[]
  }
  warning_only: boolean
  opt_in: boolean
  tags: string[]
  fix: {
    label: string
    ops: { op_id: string; params: Record<string, unknown>; mask?: Record<string, unknown> }[]
    derivation: string
  } | null
}

export interface DimensionScore {
  score: number
  severity: Severity
  breakdown: {
    metric: string
    value: number
    score: number
    weight: number
    expected: [number, number]
  }[]
  note?: string
}

export interface Scores {
  dimensions: Record<string, DimensionScore>
  overall: number
  overall_uncapped: number
  worst_dimension: string | null
  weights: Record<string, number>
}

export interface Diagnosis {
  image_id: string
  scores: Scores
  findings: Finding[]
  recipe: (StackOp & { stage: number })[]
  metrics: Record<string, Record<string, unknown>>
  document_mode: boolean
  monochrome: boolean
  notes: string[]
  maps: Record<string, RenderRef>
  strength: number
}

export interface Verification {
  overall_before: number
  overall_after: number
  overall_delta: number
  improved: boolean
  dimensions: {
    dimension: string
    before: number
    after: number
    delta: number
    severity_before: Severity
    severity_after: Severity
    expected_tradeoff: boolean
  }[]
  regressions: {
    dimension: string
    before: number
    after: number
    delta: number
    likely_cause?: { index: number; op_id: string; uid: string; recovered: number }
  }[]
  tradeoffs: { dimension: string; before: number; after: number; delta: number }[]
  metrics_moved: {
    dimension: string
    metric: string
    before: number
    after: number
    delta: number
    expected: [number, number]
    entered_range: boolean
    left_range: boolean
  }[]
  summary: string
  render?: RenderRef
}

export interface SampleInfo {
  id: string
  name: string
  blurb: string
  tags: string[]
}

export interface JobInfo {
  job_id: string
  kind: string
  status: 'pending' | 'running' | 'done' | 'error' | 'cancelled'
  progress: number
  message: string
  error: string | null
  elapsed: number
  result: unknown
}

export interface AugStage {
  op_id: string
  params: Record<string, unknown>
  p: number
}

export interface AugVariant {
  index: number
  render: RenderRef
  recipe: StackOp[]
  applied: Record<string, Record<string, unknown>>
  notes: string[]
}

export interface AugPreset {
  id: string
  name: string
  blurb: string
  stages: AugStage[]
}
