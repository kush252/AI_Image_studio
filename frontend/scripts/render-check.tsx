/**
 * Headless render check: mount the control panel for EVERY operator in the
 * live catalogue and report any that throw.
 *
 * This is the exact path that broke when a Select item used an empty-string
 * value: clicking an operator rendered its panel, Radix threw, and React
 * unmounted the whole tree, leaving a blank page. A browser is not needed to
 * catch that, only a DOM.
 *
 *   node --import tsx scripts/render-check.tsx          (server on :8000)
 *   node --import tsx scripts/render-check.tsx 8001
 */
import { JSDOM } from 'jsdom'

const PORT = process.argv[2] ?? '8000'

// ---- DOM + the browser APIs the components touch, before importing them ----
const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
  url: 'http://127.0.0.1/',
  pretendToBeVisual: true,
})
const g = globalThis as Record<string, unknown>
g.window = dom.window
g.document = dom.window.document
g.navigator = dom.window.navigator
g.HTMLElement = dom.window.HTMLElement
g.Element = dom.window.Element
g.Node = dom.window.Node
g.DocumentFragment = dom.window.DocumentFragment
g.getComputedStyle = dom.window.getComputedStyle
g.requestAnimationFrame = (cb: FrameRequestCallback) => setTimeout(() => cb(0), 0)
g.cancelAnimationFrame = (id: number) => clearTimeout(id)
g.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
}
g.DOMRect = class {
  constructor(public x = 0, public y = 0, public width = 0, public height = 0) {}
  toJSON() {
    return this
  }
}
dom.window.HTMLElement.prototype.scrollIntoView = () => {}
dom.window.HTMLElement.prototype.releasePointerCapture = () => {}
dom.window.HTMLElement.prototype.hasPointerCapture = () => false
dom.window.HTMLElement.prototype.setPointerCapture = () => {}
if (!dom.window.matchMedia) {
  // @ts-expect-error jsdom does not implement matchMedia
  dom.window.matchMedia = () => ({
    matches: false,
    addEventListener() {},
    removeEventListener() {},
  })
}

g.IS_REACT_ACT_ENVIRONMENT = true

const React = (await import('react')).default
const { createRoot } = await import('react-dom/client')
const { act } = await import('react')

const { OperatorPanel } = await import('../src/components/controls/OperatorPanel')
const { TooltipProvider } = await import('../src/components/ui')
const { useCatalogue } = await import('../src/store')
const { makeOp } = await import('../src/lib/utils')
type Catalogue = import('../src/lib/types').Catalogue

const res = await fetch(`http://127.0.0.1:${PORT}/api/operations`)
if (!res.ok) {
  console.error(`Cannot reach the API on :${PORT}. Start the backend first.`)
  process.exit(2)
}
const catalogue = (await res.json()) as Catalogue
useCatalogue.getState().setCatalogue(catalogue)

// React logs errors it re-throws; keep the output readable.
const realError = console.error
console.error = (...args: unknown[]) => {
  const first = String(args[0] ?? '')
  if (first.includes('The above error occurred') || first.includes('not wrapped in act')) return
  realError(...args)
}

const failures: { id: string; message: string }[] = []

for (const spec of catalogue.operations) {
  const host = dom.window.document.createElement('div')
  dom.window.document.body.appendChild(host)
  const root = createRoot(host)
  const op = makeOp(spec.id, spec.params)
  try {
    await act(async () => {
      root.render(
        React.createElement(
          TooltipProvider,
          null,
          React.createElement(OperatorPanel, { op }),
        ),
      )
    })
  } catch (err) {
    failures.push({
      id: spec.id,
      message: err instanceof Error ? `${err.name}: ${err.message}` : String(err),
    })
  } finally {
    try {
      await act(async () => root.unmount())
    } catch {
      /* unmount noise is not a product failure */
    }
    host.remove()
  }
}

console.error = realError
console.log(
  `\nrendered ${catalogue.operations.length} operator panels, ` +
    `${failures.length} failed`,
)
for (const f of failures) console.log(`  FAIL  ${f.id}\n        ${f.message}`)
process.exit(failures.length ? 1 : 0)
