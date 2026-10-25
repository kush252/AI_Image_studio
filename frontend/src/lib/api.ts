// API client. Preview requests carry a seq and are abortable (PLAN.md D8).
import type {
  AugPreset, AugStage, AugVariant, Catalogue, Diagnosis, ImageInfo, JobInfo,
  PreviewResponse, SampleInfo, StackOp, Verification,
} from './types'

const BASE = ''

export class ApiError extends Error {
  code: string
  field?: string
  detail?: unknown
  status: number
  constructor(status: number, code: string, message: string, field?: string, detail?: unknown) {
    super(message)
    this.status = status
    this.code = code
    this.field = field
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    let code = 'HTTP_' + res.status
    let message = res.statusText || 'Request failed'
    let field: string | undefined
    let detail: unknown
    try {
      const body = await res.json()
      if (body?.error) {
        code = body.error.code ?? code
        message = body.error.message ?? message
        field = body.error.field
        detail = body.error.detail
      }
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, code, message, field, detail)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { method: 'POST', body: JSON.stringify(body), signal })
}

// Strip client-only fields before sending an op stack to the server.
export function wireOps(ops: StackOp[]) {
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

export const api = {
  health: () => request<{ status: string; operations: number; syllabus_operations: number; uses_learned_models: boolean }>('/api/health'),

  catalogue: () => request<Catalogue>('/api/operations'),

  operation: (id: string) => request<unknown>(`/api/operations/${id}`),

  autoParams: (id: string, body: { image_id: string; ops: unknown[]; target?: string; params?: Record<string, unknown> }) =>
    post<{ op_id: string; params: Record<string, unknown> }>(`/api/operations/${id}/auto`, body),

  samples: () => request<{ samples: SampleInfo[] }>('/api/samples'),

  loadSample: (id: string) => post<ImageInfo>(`/api/samples/${id}`, {}),

  listImages: () => request<{ images: ImageInfo[] }>('/api/images'),

  deleteImage: (id: string) => request<{ deleted: string }>(`/api/images/${id}`, { method: 'DELETE' }),

  async upload(file: File): Promise<ImageInfo> {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(BASE + '/api/images', { method: 'POST', body: form })
    if (!res.ok) {
      let code = 'UPLOAD_FAILED'
      let message = 'Upload failed'
      try {
        const body = await res.json()
        if (body?.error) {
          code = body.error.code ?? code
          message = body.error.message ?? message
        }
      } catch { /* ignore */ }
      throw new ApiError(res.status, code, message)
    }
    return (await res.json()) as ImageInfo
  },

  preview: (
    body: {
      image_id: string
      ops: unknown[]
      target?: string
      seq?: number
      session?: string
      include_stats?: boolean
      include_overlays?: boolean
      format?: string
      quality?: number
    },
    signal?: AbortSignal,
  ) => post<PreviewResponse>('/api/pipeline/preview', body, signal),

  probe: (body: { image_id: string; ops: unknown[]; target?: string }, signal?: AbortSignal) =>
    post<{ stats: PreviewResponse['stats'] }>('/api/pipeline/probe', body, signal),

  cacheStats: () => request<Record<string, Record<string, number>>>('/api/pipeline/cache'),

  analysis: (body: { image_id: string; ops?: unknown[]; target?: string; groups?: string[] }) =>
    post<{ metrics: Record<string, Record<string, unknown>>; scores?: Diagnosis['scores'] }>('/api/analysis', body),

  analysisSchema: () => request<unknown>('/api/analysis/schema'),

  diagnose: (body: { image_id: string; ops?: unknown[]; target?: string; strength?: number; include_opt_in?: boolean; include_maps?: boolean }, signal?: AbortSignal) =>
    post<Diagnosis>('/api/advisor/diagnose', body, signal),

  verify: (body: { image_id: string; ops: unknown[]; target?: string; attribute_regressions?: boolean }, signal?: AbortSignal) =>
    post<Verification>('/api/advisor/verify', body, signal),

  augPreview: (body: { image_id: string; stages: AugStage[]; count: number; seed: number; one_of_groups?: Record<string, number[]> }, signal?: AbortSignal) =>
    post<{ variants: AugVariant[]; seed: number }>('/api/augment/preview', body, signal),

  augGenerate: (body: { image_ids: string[]; stages: AugStage[]; variants_per_image: number; seed: number; format?: string; quality?: number }) =>
    post<{ job_id: string; total: number }>('/api/augment/generate', body),

  augPresets: () => request<{ presets: AugPreset[] }>('/api/augment/presets'),

  export: (body: { image_id: string; ops: unknown[]; format?: string; quality?: number; bit_depth?: number; resize_long_edge?: number | null; filename?: string | null }) =>
    post<{ job_id: string }>('/api/export', body),

  batchApply: (body: { image_ids: string[]; ops: unknown[]; format?: string; quality?: number; target?: string }) =>
    post<{ job_id: string; total: number }>('/api/batch/apply', body),

  job: (id: string) => request<JobInfo>(`/api/jobs/${id}`),

  cancelJob: (id: string) => post<JobInfo>(`/api/jobs/${id}/cancel`, {}),

  report: (body: { image_id: string; ops: unknown[]; title?: string; include_advisor?: boolean; include_images?: boolean }) =>
    post<{ markdown: string; operator_count: number; syllabus_operator_count: number }>('/api/report', body),
}

/** Poll a job to completion, reporting progress. */
export async function waitForJob(
  jobId: string,
  onProgress?: (j: JobInfo) => void,
  intervalMs = 400,
): Promise<JobInfo> {
  for (;;) {
    const j = await api.job(jobId)
    onProgress?.(j)
    if (j.status === 'done' || j.status === 'error' || j.status === 'cancelled') return j
    await new Promise((r) => setTimeout(r, intervalMs))
  }
}
