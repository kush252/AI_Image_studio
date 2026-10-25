/** Preview driver. Implements PLAN.md D8: debounce AND abort, with a
 *  monotonically increasing seq so a stale response can never overwrite a
 *  newer one. */
import * as React from 'react'

import { ApiError, api, wireOps } from '@/lib/api'
import { usePipeline, useSession, useUI } from '@/store'

const SESSION_ID = Math.random().toString(36).slice(2)
const DEBOUNCE_MS = 100

export function usePreview() {
  const ops = usePipeline((s) => s.ops)
  const { setPreview, setPreviewing, setPreviewError } = usePipeline()
  const activeId = useSession((s) => s.activeId)
  const previewQuality = useUI((s) => s.previewQuality)

  const seqRef = React.useRef(0)
  const latestAppliedRef = React.useRef(0)
  const abortRef = React.useRef<AbortController | null>(null)
  const timerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)

  const run = React.useCallback(
    async (immediate = false) => {
      if (!activeId) {
        setPreview(null)
        return
      }
      if (timerRef.current) clearTimeout(timerRef.current)

      const fire = async () => {
        // Abort the in-flight request: dragging a slider must never queue up
        // dozens of renders behind each other.
        abortRef.current?.abort()
        const controller = new AbortController()
        abortRef.current = controller
        seqRef.current += 1
        const seq = seqRef.current

        setPreviewing(true)
        try {
          const res = await api.preview(
            {
              image_id: activeId,
              ops: wireOps(usePipeline.getState().ops),
              target: 'proxy',
              seq,
              session: SESSION_ID,
              include_stats: true,
              include_overlays: true,
            },
            controller.signal,
          )
          // Guard against out-of-order completion.
          if (res.seq >= latestAppliedRef.current) {
            latestAppliedRef.current = res.seq
            setPreview(res)
            setPreviewError(null)
          }
        } catch (err) {
          if (err instanceof DOMException && err.name === 'AbortError') return
          if (err instanceof ApiError) {
            // CANCELLED means the server superseded this render; not an error.
            if (err.code === 'CANCELLED') return
            setPreviewError(`${err.code}: ${err.message}`)
          } else {
            setPreviewError(err instanceof Error ? err.message : String(err))
          }
        } finally {
          if (abortRef.current === controller) setPreviewing(false)
        }
      }

      if (immediate) await fire()
      else timerRef.current = setTimeout(fire, DEBOUNCE_MS)
    },
    [activeId, setPreview, setPreviewing, setPreviewError],
  )

  // Re-render whenever the stack or the active image changes.
  React.useEffect(() => {
    void run()
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [ops, activeId, previewQuality, run])

  return { refresh: () => run(true) }
}

/** Global keyboard shortcuts (PLAN.md Phase 1). */
export function useShortcuts() {
  const { undo, redo, ops, select, selectedUid } = usePipeline()
  const { setHoldBefore, toggleLearnMode, resetView, setCompare } = useUI()

  React.useEffect(() => {
    const isTyping = (t: EventTarget | null) => {
      const el = t as HTMLElement | null
      return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
    }

    const down = (e: KeyboardEvent) => {
      if (isTyping(e.target)) return

      if (e.code === 'Space' && !e.repeat) {
        e.preventDefault()
        setHoldBefore(true)
        return
      }
      const mod = e.ctrlKey || e.metaKey
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        if (e.shiftKey) redo()
        else undo()
        return
      }
      if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault()
        redo()
        return
      }
      if (e.key === '[' || e.key === ']') {
        e.preventDefault()
        if (!ops.length) return
        const i = ops.findIndex((o) => o.uid === selectedUid)
        const next = e.key === ']'
          ? Math.min((i < 0 ? -1 : i) + 1, ops.length - 1)
          : Math.max((i < 0 ? ops.length : i) - 1, 0)
        select(ops[next].uid)
        return
      }
      if (e.key === 'l' && !mod) toggleLearnMode()
      if (e.key === '0' && !mod) resetView()
      if (e.key === 'c' && !mod) setCompare('split')
      if (e.key === 'v' && !mod) setCompare('single')
      if (e.key === 'b' && !mod) setCompare('side')
    }

    const up = (e: KeyboardEvent) => {
      if (e.code === 'Space') setHoldBefore(false)
    }

    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [undo, redo, ops, selectedUid, select, setHoldBefore, toggleLearnMode, resetView, setCompare])
}
