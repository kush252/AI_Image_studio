/** The operator stack: reorder, toggle, duplicate, mute-below, select. */
import {
  ArrowDownToLine, Copy, Eye, EyeOff, GripVertical, Layers, Trash2, X,
} from 'lucide-react'
import * as React from 'react'

import { Badge, Button, EmptyState, Tooltip } from '@/components/ui'
import type { StackOp } from '@/lib/types'
import { cn, fmtMs } from '@/lib/utils'
import { useCatalogue, usePipeline } from '@/store'

export function OpStack() {
  const {
    ops, selectedUid, select, removeOp, toggleOp, duplicateOp, moveOp,
    muteBelow, clear, preview,
  } = usePipeline()
  const byId = useCatalogue((s) => s.byId)
  const [dragFrom, setDragFrom] = React.useState<number | null>(null)
  const [dragOver, setDragOver] = React.useState<number | null>(null)

  const timings = React.useMemo(() => {
    const map = new Map<string, { ms: number; cached: boolean }>()
    preview?.timings.forEach((t, i) => {
      const op = ops[i]
      if (op) map.set(op.uid, { ms: t.ms, cached: t.cached })
    })
    return map
  }, [preview, ops])

  if (!ops.length) {
    return (
      <EmptyState icon={<Layers className="h-7 w-7" />} title="No operators yet" className="py-6">
        Pick an operator from the left to start building a pipeline. Nothing is
        applied destructively: the stack is recomputed from the original every
        time.
      </EmptyState>
    )
  }

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between px-1 pb-1">
        <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
          Pipeline ({ops.length})
        </span>
        <div className="flex items-center gap-1">
          {preview && (
            <span className="text-[10px] tabular text-muted-foreground">
              {fmtMs(preview.total_ms)}
              {preview.cache_hits > 0 && (
                <span className="ml-1 text-primary/70">{preview.cache_hits} cached</span>
              )}
            </span>
          )}
          <Tooltip content="Remove all operators">
            <Button variant="ghost" size="icon-sm" onClick={clear} aria-label="Clear pipeline">
              <X className="h-3 w-3" />
            </Button>
          </Tooltip>
        </div>
      </div>

      {ops.map((op, i) => (
        <OpCard
          key={op.uid}
          op={op}
          index={i}
          total={ops.length}
          selected={selectedUid === op.uid}
          name={byId[op.op_id]?.name ?? op.op_id}
          syllabus={byId[op.op_id]?.syllabus ?? false}
          isView={byId[op.op_id]?.is_view ?? false}
          timing={timings.get(op.uid)}
          dragOver={dragOver === i}
          onSelect={() => select(op.uid === selectedUid ? null : op.uid)}
          onToggle={() => toggleOp(op.uid)}
          onRemove={() => removeOp(op.uid)}
          onDuplicate={() => duplicateOp(op.uid)}
          onMuteBelow={() => muteBelow(i)}
          onDragStart={() => setDragFrom(i)}
          onDragOver={() => setDragOver(i)}
          onDrop={() => {
            if (dragFrom !== null && dragFrom !== i) moveOp(dragFrom, i)
            setDragFrom(null)
            setDragOver(null)
          }}
          onDragEnd={() => {
            setDragFrom(null)
            setDragOver(null)
          }}
        />
      ))}

      {preview?.notes && preview.notes.length > 0 && (
        <div className="mt-2 space-y-1 rounded border border-sev-moderate/40 bg-sev-moderate/10 p-2">
          {preview.notes.map((n, i) => (
            <p key={i} className="text-[10px] leading-snug text-sev-moderate">
              {n}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

function OpCard({
  op, index, total, selected, name, syllabus, isView, timing, dragOver,
  onSelect, onToggle, onRemove, onDuplicate, onMuteBelow,
  onDragStart, onDragOver, onDrop, onDragEnd,
}: {
  op: StackOp
  index: number
  total: number
  selected: boolean
  name: string
  syllabus: boolean
  isView: boolean
  timing?: { ms: number; cached: boolean }
  dragOver: boolean
  onSelect: () => void
  onToggle: () => void
  onRemove: () => void
  onDuplicate: () => void
  onMuteBelow: () => void
  onDragStart: () => void
  onDragOver: () => void
  onDrop: () => void
  onDragEnd: () => void
}) {
  return (
    <div
      draggable
      onDragStart={onDragStart}
      onDragOver={(e) => {
        e.preventDefault()
        onDragOver()
      }}
      onDrop={(e) => {
        e.preventDefault()
        onDrop()
      }}
      onDragEnd={onDragEnd}
      className={cn(
        'group flex items-center gap-1 rounded border px-1.5 py-1 transition-colors',
        selected ? 'border-primary/60 bg-primary/10' : 'border-border/60 bg-card hover:bg-accent/40',
        !op.enabled && 'opacity-45',
        dragOver && 'border-primary border-dashed',
      )}
    >
      <GripVertical className="h-3 w-3 shrink-0 cursor-grab text-muted-foreground/50" />

      <span className="w-4 shrink-0 text-[10px] tabular text-muted-foreground">{index + 1}</span>

      <button onClick={onSelect} className="min-w-0 flex-1 text-left">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-xs font-medium">{op.name ?? name}</span>
          {syllabus && <Badge variant="syllabus">S</Badge>}
          {isView && (
            <Tooltip content="A view: reports numbers without altering the image.">
              <Badge variant="outline">view</Badge>
            </Tooltip>
          )}
          {op.from_finding && (
            <Tooltip content={`Added by the Advisor from finding ${op.from_finding}`}>
              <Badge variant="default">advisor</Badge>
            </Tooltip>
          )}
        </div>
        <div className="flex items-center gap-1.5 text-[9px] text-muted-foreground">
          {op.blend !== 'normal' && <span>{op.blend}</span>}
          {op.opacity < 1 && <span>{Math.round(op.opacity * 100)}%</span>}
          {op.mask && <span className="text-primary/70">masked</span>}
          {timing && (
            <span className={cn('tabular', timing.cached && 'text-primary/60')}>
              {timing.cached ? 'cached' : fmtMs(timing.ms)}
            </span>
          )}
        </div>
      </button>

      <div className="flex shrink-0 items-center opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
        {index < total - 1 && (
          <Tooltip content="Disable everything below this">
            <Button variant="ghost" size="icon-sm" onClick={onMuteBelow} aria-label="Mute below">
              <ArrowDownToLine className="h-3 w-3" />
            </Button>
          </Tooltip>
        )}
        <Tooltip content="Duplicate">
          <Button variant="ghost" size="icon-sm" onClick={onDuplicate} aria-label="Duplicate operator">
            <Copy className="h-3 w-3" />
          </Button>
        </Tooltip>
        <Tooltip content="Remove">
          <Button variant="ghost" size="icon-sm" onClick={onRemove} aria-label="Remove operator">
            <Trash2 className="h-3 w-3" />
          </Button>
        </Tooltip>
      </div>

      <Tooltip content={op.enabled ? 'Disable' : 'Enable'}>
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onToggle}
          aria-label={op.enabled ? 'Disable operator' : 'Enable operator'}
          aria-pressed={op.enabled}
        >
          {op.enabled ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
        </Button>
      </Tooltip>
    </div>
  )
}
