/** App shell: left rail with the two (plus one) sections, image bar, and the
 *  status bar. PLAN.md Section 3. */
import {
  Activity, BookOpen, Image as ImageIcon, Moon, Shuffle, Sliders, Sun, Trash2,
  Upload, X,
} from 'lucide-react'
import * as React from 'react'

import { Badge, Button, Spinner, Tooltip } from '@/components/ui'
import { api } from '@/lib/api'
import { cn, fmtBytes, fmtMs } from '@/lib/utils'
import { useCatalogue, usePipeline, useSession, useUI } from '@/store'

const SECTIONS = [
  { id: 'studio', label: 'Studio', icon: Sliders, blurb: 'Manual editor: every operator, under your control.' },
  { id: 'advisor', label: 'Advisor', icon: Activity, blurb: 'Measure the image and get derived, one-click fixes.' },
  { id: 'augment', label: 'Augment', icon: Shuffle, blurb: 'Generate randomised variants for a dataset.' },
] as const

export function LeftRail() {
  const { section, setSection, theme, toggleTheme, learnMode, toggleLearnMode } = useUI()
  const catalogue = useCatalogue((s) => s.catalogue)

  return (
    <nav className="flex w-[66px] shrink-0 flex-col items-center gap-1 border-r border-border bg-card py-2">
      <div className="mb-1 flex h-8 w-8 items-center justify-center rounded-md bg-primary/15">
        <svg viewBox="0 0 32 32" className="h-5 w-5" aria-hidden>
          <path d="M7 22l5.5-7 4 5 3-3.5L25 22H7z" fill="hsl(var(--primary))" />
          <circle cx="21" cy="10.5" r="2.5" fill="hsl(var(--primary))" />
        </svg>
      </div>

      {SECTIONS.map((s) => {
        const Icon = s.icon
        const active = section === s.id
        return (
          <Tooltip key={s.id} content={<div><strong>{s.label}</strong><br />{s.blurb}</div>} side="right">
            <button
              onClick={() => setSection(s.id)}
              className={cn(
                'flex w-[54px] flex-col items-center gap-0.5 rounded-md py-1.5 transition-colors',
                active ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:bg-accent hover:text-foreground',
              )}
              aria-current={active ? 'page' : undefined}
            >
              <Icon className="h-[18px] w-[18px]" />
              <span className="text-[9px] font-medium">{s.label}</span>
            </button>
          </Tooltip>
        )
      })}

      <div className="flex-1" />

      <Tooltip content={learnMode ? 'Learn mode on: formulas and theory shown' : 'Learn mode off'} side="right">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={toggleLearnMode}
          className={cn(learnMode && 'text-primary')}
          aria-label="Toggle learn mode"
          aria-pressed={learnMode}
        >
          <BookOpen className="h-4 w-4" />
        </Button>
      </Tooltip>
      <Tooltip content="Toggle theme" side="right">
        <Button variant="ghost" size="icon-sm" onClick={toggleTheme} aria-label="Toggle theme">
          {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>
      </Tooltip>
      {catalogue && (
        <Tooltip
          content={`${catalogue.total} operators, ${catalogue.syllabus_count} of them syllabus topics. No learned models.`}
          side="right"
        >
          <span className="cursor-help pb-1 text-[9px] tabular text-muted-foreground">
            {catalogue.total}
          </span>
        </Tooltip>
      )}
    </nav>
  )
}

export function ImageBar() {
  const { images, activeId, setActive, addImage, removeImage } = useSession()
  const [busy, setBusy] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const fileRef = React.useRef<HTMLInputElement>(null)

  const upload = async (files: FileList | File[]) => {
    setBusy(true)
    setError(null)
    try {
      for (const f of Array.from(files)) {
        const info = await api.upload(f)
        addImage(info)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  // Paste an image straight from the clipboard.
  React.useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = Array.from(e.clipboardData?.items ?? [])
      const files = items
        .filter((i) => i.kind === 'file' && i.type.startsWith('image/'))
        .map((i) => i.getAsFile())
        .filter((f): f is File => !!f)
      if (files.length) void upload(files)
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  if (!images.length) return null

  return (
    <div className="flex h-[52px] shrink-0 items-center gap-1.5 overflow-x-auto border-b border-border bg-card px-2 scrollbar-thin">
      {images.map((img) => (
        <button
          key={img.image_id}
          onClick={() => setActive(img.image_id)}
          className={cn(
            'group relative h-10 w-14 shrink-0 overflow-hidden rounded border transition-colors',
            activeId === img.image_id ? 'border-primary' : 'border-border hover:border-muted-foreground',
          )}
          title={`${img.name} — ${img.width}x${img.height}`}
        >
          {img.thumbnail && (
            <img src={img.thumbnail.url} alt={img.name} className="h-full w-full object-cover" />
          )}
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => {
              e.stopPropagation()
              void api.deleteImage(img.image_id).catch(() => undefined)
              removeImage(img.image_id)
            }}
            onKeyDown={(e) => { if (e.key === 'Enter') e.stopPropagation() }}
            className="absolute right-0 top-0 hidden rounded-bl bg-background/90 p-0.5 group-hover:block"
            aria-label={`Remove ${img.name}`}
          >
            <X className="h-2.5 w-2.5" />
          </span>
        </button>
      ))}

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => e.target.files && upload(e.target.files)}
      />
      <Tooltip content="Add another image (or paste from the clipboard)">
        <Button variant="outline" size="icon-sm" onClick={() => fileRef.current?.click()} disabled={busy} aria-label="Add image">
          {busy ? <Spinner /> : <Upload className="h-3.5 w-3.5" />}
        </Button>
      </Tooltip>

      {error && <span className="text-[10px] text-destructive">{error}</span>}
    </div>
  )
}

export function StatusBar() {
  const active = useSession((s) => s.images.find((i) => i.image_id === s.activeId) ?? null)
  const { preview, previewing, previewError, ops } = usePipeline()
  const { previewQuality, setPreviewQuality } = useUI()

  return (
    <footer className="flex h-7 shrink-0 items-center gap-3 border-t border-border bg-card px-3 text-[10px] tabular text-muted-foreground">
      {active ? (
        <>
          <span className="truncate font-medium text-foreground/80">{active.name}</span>
          <span>
            {active.width}×{active.height}
          </span>
          <span>{active.megapixels} MP</span>
          {active.proxy_scale < 1 && (
            <Tooltip content="Previews render on a downscaled proxy for speed; export runs at full resolution with scale-aware parameters.">
              <span className="cursor-help text-primary/70">
                proxy {Math.round(active.proxy_scale * 100)}%
              </span>
            </Tooltip>
          )}
          <span>
            {active.source_format.toUpperCase()} {active.source_bit_depth}-bit
          </span>
          {active.was_grayscale && <Badge variant="outline">grayscale</Badge>}
          {active.had_alpha && <Badge variant="outline">alpha</Badge>}
        </>
      ) : (
        <span>No image loaded</span>
      )}

      <div className="flex-1" />

      <span>{ops.length} op{ops.length === 1 ? '' : 's'}</span>
      {preview && (
        <>
          <span>{fmtMs(preview.total_ms)}</span>
          <span>{fmtBytes(preview.render.bytes)}</span>
        </>
      )}
      {previewing && <Spinner className="h-2.5 w-2.5" />}
      {previewError && (
        <span className="max-w-[320px] truncate text-destructive" title={previewError}>
          {previewError}
        </span>
      )}

      <Tooltip content="Preview resolution. Lower is faster; export is always full resolution.">
        <select
          value={previewQuality}
          onChange={(e) => setPreviewQuality(Number(e.target.value))}
          className="cursor-pointer rounded border border-input bg-background px-1 py-0 text-[10px]"
          aria-label="Preview resolution"
        >
          <option value={1024}>1024 px</option>
          <option value={1600}>1600 px</option>
          <option value={2048}>2048 px</option>
        </select>
      </Tooltip>
    </footer>
  )
}

export function Dropzone() {
  const { addImage } = useSession()
  const [samples, setSamples] = React.useState<{ id: string; name: string; blurb: string; tags: string[] }[]>([])
  const [busy, setBusy] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [over, setOver] = React.useState(false)
  const fileRef = React.useRef<HTMLInputElement>(null)

  React.useEffect(() => {
    api.samples().then((r) => setSamples(r.samples)).catch(() => undefined)
  }, [])

  const upload = async (files: FileList | File[]) => {
    setBusy('upload')
    setError(null)
    try {
      for (const f of Array.from(files)) addImage(await api.upload(f))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  const loadSample = async (id: string) => {
    setBusy(id)
    setError(null)
    try {
      addImage(await api.loadSample(id))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="flex h-full items-center justify-center overflow-y-auto p-6 scrollbar-thin">
      <div className="w-full max-w-2xl space-y-5">
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setOver(true)
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault()
            setOver(false)
            if (e.dataTransfer.files.length) void upload(e.dataTransfer.files)
          }}
          onClick={() => fileRef.current?.click()}
          className={cn(
            'cursor-pointer rounded-lg border-2 border-dashed p-10 text-center transition-colors',
            over ? 'border-primary bg-primary/5' : 'border-border hover:border-muted-foreground',
          )}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => e.target.files && upload(e.target.files)}
          />
          {busy === 'upload' ? (
            <Spinner className="mx-auto h-6 w-6" />
          ) : (
            <ImageIcon className="mx-auto h-8 w-8 text-muted-foreground/60" />
          )}
          <p className="mt-2.5 text-sm font-medium">Drop an image, or click to browse</p>
          <p className="mt-1 text-xs text-muted-foreground">
            JPEG, PNG, WebP, TIFF or BMP · up to 40 megapixels · you can also paste
            from the clipboard
          </p>
        </div>

        {error && (
          <div className="rounded border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {error}
          </div>
        )}

        <div>
          <div className="mb-2 flex items-center gap-2">
            <span className="text-xs font-semibold">Or start from a sample</span>
            <span className="text-[10px] text-muted-foreground">
              each one exercises a different part of the platform
            </span>
          </div>
          <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {samples.map((s) => (
              <button
                key={s.id}
                onClick={() => loadSample(s.id)}
                disabled={busy !== null}
                className="flex items-start gap-2 rounded border border-border bg-card p-2.5 text-left transition-colors hover:border-primary/50 hover:bg-accent/40 disabled:opacity-50"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="truncate text-xs font-medium">{s.name}</span>
                    {busy === s.id && <Spinner className="h-2.5 w-2.5" />}
                  </div>
                  <p className="mt-0.5 text-[10px] leading-snug text-muted-foreground">{s.blurb}</p>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export function ImageManager() {
  const { images, removeImage } = useSession()
  if (images.length < 2) return null
  return (
    <div className="space-y-1">
      {images.map((i) => (
        <div key={i.image_id} className="flex items-center gap-2 rounded px-1 py-0.5 text-[10px]">
          <span className="truncate">{i.name}</span>
          <span className="text-muted-foreground">{i.width}×{i.height}</span>
          <div className="flex-1" />
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => {
              void api.deleteImage(i.image_id).catch(() => undefined)
              removeImage(i.image_id)
            }}
            aria-label={`Remove ${i.name}`}
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        </div>
      ))}
    </div>
  )
}
