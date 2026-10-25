import * as React from 'react'

import { ErrorBoundary } from '@/components/ErrorBoundary'
import { ImageBar, LeftRail, StatusBar } from '@/components/layout/Shell'
import { Spinner, TooltipProvider } from '@/components/ui'
import { Advisor } from '@/features/advisor/Advisor'
import { Augment } from '@/features/augment/Augment'
import { Studio } from '@/features/studio/Studio'
import { api } from '@/lib/api'
import { usePreview, useShortcuts } from '@/lib/usePreview'
import { useCatalogue, useUI } from '@/store'

export default function App() {
  const section = useUI((s) => s.section)
  const { catalogue, setCatalogue } = useCatalogue()
  const [error, setError] = React.useState<string | null>(null)

  // The preview driver and shortcuts live at the root so they stay mounted
  // across section changes: switching to the Advisor must not cancel a render.
  usePreview()
  useShortcuts()

  React.useEffect(() => {
    api
      .catalogue()
      .then(setCatalogue)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [setCatalogue])

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="max-w-md space-y-2 rounded-lg border border-destructive/40 bg-destructive/10 p-5">
          <h1 className="text-sm font-semibold text-destructive">
            Cannot reach the backend
          </h1>
          <p className="text-xs leading-relaxed text-muted-foreground">{error}</p>
          <p className="text-xs leading-relaxed text-muted-foreground">
            Start it with{' '}
            <code className="rounded bg-muted px-1 py-0.5">
              uvicorn app.main:app --reload
            </code>{' '}
            from the <code className="rounded bg-muted px-1 py-0.5">backend</code>{' '}
            directory, then reload this page.
          </p>
        </div>
      </div>
    )
  }

  if (!catalogue) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3">
        <Spinner className="h-6 w-6" />
        <span className="text-xs text-muted-foreground">Loading the operator catalogue...</span>
      </div>
    )
  }

  return (
    <TooltipProvider delayDuration={350} skipDelayDuration={200}>
      <div className="flex h-full overflow-hidden">
        <LeftRail />
        <div className="flex min-w-0 flex-1 flex-col">
          <ImageBar />
          <ErrorBoundary
            key={section}
            label={`The ${section} section hit an error`}
          >
            {section === 'studio' && <Studio />}
            {section === 'advisor' && <Advisor />}
            {section === 'augment' && <Augment />}
          </ErrorBoundary>
          <StatusBar />
        </div>
      </div>
    </TooltipProvider>
  )
}
