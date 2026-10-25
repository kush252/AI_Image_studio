/** Catches render errors so a crash shows what went wrong instead of a blank
 *  page. Without this, any thrown error unmounts the whole tree and the dark
 *  theme leaves you looking at a black screen with no clue what happened. */
import { AlertTriangle, RotateCcw } from 'lucide-react'
import * as React from 'react'

interface State {
  error: Error | null
  info: string | null
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode; label?: string },
  State
> {
  state: State = { error: null, info: null }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Keep it in the console too, with the component stack, which is usually
    // the fastest way to find the offending component.
    console.error('Render error:', error, info.componentStack)
    this.setState({ info: info.componentStack ?? null })
  }

  render() {
    const { error, info } = this.state
    if (!error) return this.props.children

    return (
      <div className="flex h-full min-h-[200px] items-center justify-center overflow-auto p-5">
        <div className="max-w-xl space-y-3 rounded-lg border border-destructive/40 bg-destructive/10 p-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-destructive" />
            <h2 className="text-sm font-semibold text-destructive">
              {this.props.label ?? 'Something broke here'}
            </h2>
          </div>

          <p className="text-xs leading-relaxed">
            <code className="rounded bg-background/60 px-1 py-0.5">
              {error.name}: {error.message}
            </code>
          </p>

          {info && (
            <details className="text-[10px] text-muted-foreground">
              <summary className="cursor-pointer select-none">
                Component stack
              </summary>
              <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-background/60 p-2 scrollbar-thin">
                {info.trim()}
              </pre>
            </details>
          )}

          <div className="flex gap-2">
            <button
              onClick={() => this.setState({ error: null, info: null })}
              className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Try again
            </button>
            <button
              onClick={() => window.location.reload()}
              className="rounded-md border border-input px-3 py-1.5 text-xs font-medium hover:bg-accent"
            >
              Reload the page
            </button>
          </div>
        </div>
      </div>
    )
  }
}
