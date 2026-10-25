import * as React from 'react'
import ReactDOM from 'react-dom/client'

import App from './App'
import './index.css'

// Honour the OS preference on first load, unless the user has already chosen.
const stored = localStorage.getItem('ipcv-theme')
if (stored === 'light' || stored === 'dark') {
  document.documentElement.dataset.theme = stored
} else if (window.matchMedia?.('(prefers-color-scheme: light)').matches) {
  document.documentElement.dataset.theme = 'light'
}

// Persist whatever the toggle sets, so a reload keeps it.
const observer = new MutationObserver(() => {
  const t = document.documentElement.dataset.theme
  if (t) {
    try {
      localStorage.setItem('ipcv-theme', t)
    } catch {
      // Private browsing can refuse storage; the theme still works for the session.
    }
  }
})
observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
