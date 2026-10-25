// Node has no idea what a .css import means; Vite does. For headless render
// checks we only need the JS, so resolve CSS to an empty module.
export async function load(url, context, nextLoad) {
  if (url.endsWith('.css')) {
    return { format: 'module', shortCircuit: true, source: 'export default {}' }
  }
  return nextLoad(url, context)
}
