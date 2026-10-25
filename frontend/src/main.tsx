import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/source-serif-4/wght.css'
import '@fontsource-variable/source-serif-4/wght-italic.css'
import '@fontsource-variable/inter/wght.css'
import '@fontsource/tiro-devanagari-hindi/devanagari-400.css'
import './styles/tokens.css'
import './styles/base.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
