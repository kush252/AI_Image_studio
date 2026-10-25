import { create } from 'zustand'

import type {
  Catalogue, Diagnosis, ImageInfo, OperationSpec, PreviewResponse, StackOp,
  Verification,
} from '@/lib/types'
import { uid } from '@/lib/utils'

/* ------------------------------------------------------------------ catalogue */

interface CatalogueState {
  catalogue: Catalogue | null
  byId: Record<string, OperationSpec>
  setCatalogue: (c: Catalogue) => void
}

export const useCatalogue = create<CatalogueState>((set) => ({
  catalogue: null,
  byId: {},
  setCatalogue: (c) =>
    set({
      catalogue: c,
      byId: Object.fromEntries(c.operations.map((o) => [o.id, o])),
    }),
}))

/* -------------------------------------------------------------------- session */

interface SessionState {
  images: ImageInfo[]
  activeId: string | null
  addImage: (img: ImageInfo) => void
  removeImage: (id: string) => void
  setActive: (id: string | null) => void
  active: () => ImageInfo | null
}

export const useSession = create<SessionState>((set, get) => ({
  images: [],
  activeId: null,
  addImage: (img) =>
    set((s) => ({
      images: [img, ...s.images.filter((i) => i.image_id !== img.image_id)],
      activeId: img.image_id,
    })),
  removeImage: (id) =>
    set((s) => {
      const images = s.images.filter((i) => i.image_id !== id)
      return {
        images,
        activeId: s.activeId === id ? (images[0]?.image_id ?? null) : s.activeId,
      }
    }),
  setActive: (id) => set({ activeId: id }),
  active: () => {
    const s = get()
    return s.images.find((i) => i.image_id === s.activeId) ?? null
  },
}))

/* ------------------------------------------------------- pipeline + history */

export interface Snapshot {
  id: string
  name: string
  ops: StackOp[]
  renderUrl?: string
  createdAt: number
}

interface PipelineState {
  ops: StackOp[]
  selectedUid: string | null
  past: StackOp[][]
  future: StackOp[][]
  snapshots: Snapshot[]

  preview: PreviewResponse | null
  previewing: boolean
  previewError: string | null

  diagnosis: Diagnosis | null
  diagnosing: boolean
  verification: Verification | null
  dismissed: string[]

  /** Push the current stack onto the undo trail. Call BEFORE mutating. */
  commit: () => void
  setOps: (ops: StackOp[], recordHistory?: boolean) => void
  addOp: (op: StackOp, index?: number) => void
  removeOp: (uid: string) => void
  updateParams: (uid: string, params: Record<string, unknown>) => void
  patchOp: (uid: string, patch: Partial<StackOp>) => void
  moveOp: (from: number, to: number) => void
  duplicateOp: (uid: string) => void
  toggleOp: (uid: string) => void
  muteBelow: (index: number) => void
  clear: () => void
  select: (uid: string | null) => void

  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean

  addSnapshot: (name: string, renderUrl?: string) => void
  removeSnapshot: (id: string) => void
  restoreSnapshot: (id: string) => void

  setPreview: (p: PreviewResponse | null) => void
  setPreviewing: (v: boolean) => void
  setPreviewError: (e: string | null) => void
  setDiagnosis: (d: Diagnosis | null) => void
  setDiagnosing: (v: boolean) => void
  setVerification: (v: Verification | null) => void
  dismissFinding: (id: string) => void
  resetDismissed: () => void
}

const HISTORY_LIMIT = 100

export const usePipeline = create<PipelineState>((set, get) => ({
  ops: [],
  selectedUid: null,
  past: [],
  future: [],
  snapshots: [],

  preview: null,
  previewing: false,
  previewError: null,

  diagnosis: null,
  diagnosing: false,
  verification: null,
  dismissed: [],

  commit: () =>
    set((s) => ({
      past: [...s.past, s.ops].slice(-HISTORY_LIMIT),
      future: [],
    })),

  setOps: (ops, recordHistory = true) =>
    set((s) => ({
      ops,
      past: recordHistory ? [...s.past, s.ops].slice(-HISTORY_LIMIT) : s.past,
      future: recordHistory ? [] : s.future,
    })),

  addOp: (op, index) =>
    set((s) => {
      const ops = [...s.ops]
      ops.splice(index ?? ops.length, 0, op)
      return {
        ops,
        selectedUid: op.uid,
        past: [...s.past, s.ops].slice(-HISTORY_LIMIT),
        future: [],
      }
    }),

  removeOp: (opUid) =>
    set((s) => ({
      ops: s.ops.filter((o) => o.uid !== opUid),
      selectedUid: s.selectedUid === opUid ? null : s.selectedUid,
      past: [...s.past, s.ops].slice(-HISTORY_LIMIT),
      future: [],
    })),

  // Param edits are coalesced: dragging a slider must not fill the undo trail
  // with one entry per frame, so history is only recorded on the first edit
  // of a run (tracked by the caller via commit()).
  updateParams: (opUid, params) =>
    set((s) => ({
      ops: s.ops.map((o) =>
        o.uid === opUid ? { ...o, params: { ...o.params, ...params } } : o,
      ),
    })),

  patchOp: (opUid, patch) =>
    set((s) => ({
      ops: s.ops.map((o) => (o.uid === opUid ? { ...o, ...patch } : o)),
    })),

  moveOp: (from, to) =>
    set((s) => {
      if (from === to || from < 0 || to < 0 || from >= s.ops.length || to >= s.ops.length) {
        return {}
      }
      const ops = [...s.ops]
      const [moved] = ops.splice(from, 1)
      ops.splice(to, 0, moved)
      return { ops, past: [...s.past, s.ops].slice(-HISTORY_LIMIT), future: [] }
    }),

  duplicateOp: (opUid) =>
    set((s) => {
      const i = s.ops.findIndex((o) => o.uid === opUid)
      if (i < 0) return {}
      const copy: StackOp = { ...s.ops[i], uid: uid(), params: { ...s.ops[i].params } }
      const ops = [...s.ops]
      ops.splice(i + 1, 0, copy)
      return {
        ops, selectedUid: copy.uid,
        past: [...s.past, s.ops].slice(-HISTORY_LIMIT), future: [],
      }
    }),

  toggleOp: (opUid) =>
    set((s) => ({
      ops: s.ops.map((o) => (o.uid === opUid ? { ...o, enabled: !o.enabled } : o)),
      past: [...s.past, s.ops].slice(-HISTORY_LIMIT),
      future: [],
    })),

  muteBelow: (index) =>
    set((s) => ({
      ops: s.ops.map((o, i) => (i > index ? { ...o, enabled: false } : o)),
      past: [...s.past, s.ops].slice(-HISTORY_LIMIT),
      future: [],
    })),

  clear: () =>
    set((s) => ({
      ops: [], selectedUid: null,
      past: [...s.past, s.ops].slice(-HISTORY_LIMIT), future: [],
      verification: null,
    })),

  select: (opUid) => set({ selectedUid: opUid }),

  undo: () =>
    set((s) => {
      if (!s.past.length) return {}
      const prev = s.past[s.past.length - 1]
      return {
        ops: prev,
        past: s.past.slice(0, -1),
        future: [s.ops, ...s.future].slice(0, HISTORY_LIMIT),
      }
    }),

  redo: () =>
    set((s) => {
      if (!s.future.length) return {}
      const next = s.future[0]
      return {
        ops: next,
        past: [...s.past, s.ops].slice(-HISTORY_LIMIT),
        future: s.future.slice(1),
      }
    }),

  canUndo: () => get().past.length > 0,
  canRedo: () => get().future.length > 0,

  addSnapshot: (name, renderUrl) =>
    set((s) => ({
      snapshots: [
        { id: uid('snap'), name, ops: s.ops.map((o) => ({ ...o })), renderUrl, createdAt: Date.now() },
        ...s.snapshots,
      ].slice(0, 24),
    })),

  removeSnapshot: (id) => set((s) => ({ snapshots: s.snapshots.filter((x) => x.id !== id) })),

  restoreSnapshot: (id) =>
    set((s) => {
      const snap = s.snapshots.find((x) => x.id === id)
      if (!snap) return {}
      return {
        ops: snap.ops.map((o) => ({ ...o })),
        past: [...s.past, s.ops].slice(-HISTORY_LIMIT),
        future: [],
      }
    }),

  setPreview: (p) => set({ preview: p }),
  setPreviewing: (v) => set({ previewing: v }),
  setPreviewError: (e) => set({ previewError: e }),
  setDiagnosis: (d) => set({ diagnosis: d }),
  setDiagnosing: (v) => set({ diagnosing: v }),
  setVerification: (v) => set({ verification: v }),
  dismissFinding: (id) => set((s) => ({ dismissed: [...s.dismissed, id] })),
  resetDismissed: () => set({ dismissed: [] }),
}))

/* ------------------------------------------------------------------------ UI */

export type Section = 'studio' | 'advisor' | 'augment'
export type CompareMode = 'single' | 'split' | 'side' | 'difference'

interface UIState {
  section: Section
  theme: 'dark' | 'light'
  compare: CompareMode
  splitPosition: number
  holdBefore: boolean
  zoom: number
  fit: boolean
  pan: { x: number; y: number }
  showOverlay: string | null
  activeCategory: string
  search: string
  learnMode: boolean
  showAdvanced: boolean
  inspectorTab: 'params' | 'histogram' | 'metrics' | 'explain'
  previewQuality: number

  setSection: (s: Section) => void
  toggleTheme: () => void
  setCompare: (c: CompareMode) => void
  setSplitPosition: (v: number) => void
  setHoldBefore: (v: boolean) => void
  setZoom: (z: number) => void
  setFit: (v: boolean) => void
  setPan: (p: { x: number; y: number }) => void
  resetView: () => void
  setShowOverlay: (name: string | null) => void
  setActiveCategory: (c: string) => void
  setSearch: (s: string) => void
  toggleLearnMode: () => void
  toggleAdvanced: () => void
  setInspectorTab: (t: UIState['inspectorTab']) => void
  setPreviewQuality: (n: number) => void
}

export const useUI = create<UIState>((set) => ({
  section: 'studio',
  theme: 'dark',
  compare: 'single',
  splitPosition: 0.5,
  holdBefore: false,
  zoom: 1,
  fit: true,
  pan: { x: 0, y: 0 },
  showOverlay: null,
  activeCategory: 'tone',
  search: '',
  learnMode: true,
  showAdvanced: false,
  inspectorTab: 'params',
  previewQuality: 1600,

  setSection: (s) => set({ section: s }),
  toggleTheme: () =>
    set((s) => {
      const theme = s.theme === 'dark' ? 'light' : 'dark'
      document.documentElement.dataset.theme = theme
      return { theme }
    }),
  setCompare: (c) => set({ compare: c }),
  setSplitPosition: (v) => set({ splitPosition: Math.min(Math.max(v, 0), 1) }),
  setHoldBefore: (v) => set({ holdBefore: v }),
  setZoom: (z) => set({ zoom: Math.min(Math.max(z, 0.05), 32), fit: false }),
  setFit: (v) => set({ fit: v }),
  setPan: (p) => set({ pan: p }),
  resetView: () => set({ zoom: 1, pan: { x: 0, y: 0 }, fit: true }),
  setShowOverlay: (name) => set({ showOverlay: name }),
  setActiveCategory: (c) => set({ activeCategory: c }),
  setSearch: (s) => set({ search: s }),
  toggleLearnMode: () => set((s) => ({ learnMode: !s.learnMode })),
  toggleAdvanced: () => set((s) => ({ showAdvanced: !s.showAdvanced })),
  setInspectorTab: (t) => set({ inspectorTab: t }),
  setPreviewQuality: (n) => set({ previewQuality: n }),
}))
