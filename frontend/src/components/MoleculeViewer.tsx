import { useState, useEffect } from 'react'
import { Atom } from 'lucide-react'

interface MoleculeViewerProps {
  smiles?: string
  name: string
  height?: string // e.g. 'h-[420px]' or 'h-[500px]'
}

export function MoleculeViewer({ smiles, name, height = 'h-[500px]' }: MoleculeViewerProps) {
  const [imgError, setImgError] = useState(false)
  const [isModelDark, setIsModelDark] = useState(() =>
    document.documentElement.classList.contains('dark-mode')
  )

  // Reset error when smiles changes
  useEffect(() => {
    setImgError(false)
  }, [smiles])

  // Sync with global dark mode class mutations
  useEffect(() => {
    const checkDark = () =>
      setIsModelDark(document.documentElement.classList.contains('dark-mode'))
    checkDark()
    const observer = new MutationObserver(checkDark)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])

  if (!smiles || imgError) {
    return (
      <div className={`w-full ${height} flex items-center justify-center bg-[var(--bg-app)] rounded border border-[#E2DED8] dark:border-[#2C2C2E] p-6`}>
        <div className="text-center space-y-3">
          <Atom className="w-10 h-10 text-[var(--text-secondary)] mx-auto animate-pulse" />
          <p className="text-xs text-[var(--text-secondary)] font-medium">
            {smiles ? 'Structure rendering failed' : 'SMILES structure not available'}<br />for {name}
          </p>
        </div>
      </div>
    )
  }

  const imgUrl = `https://cactus.nci.nih.gov/chemical/structure/${encodeURIComponent(smiles)}/image?format=gif&width=800&height=800`

  return (
    <div
      className={`w-full ${height} rounded overflow-hidden border border-[#E2DED8] dark:border-[#2C2C2E] p-6 flex items-center justify-center relative transition-colors duration-300`}
      style={{ backgroundColor: isModelDark ? '#111111' : '#FFFFFF' }}
    >
      <img
        src={imgUrl}
        alt={`2D structure of ${name}`}
        className={`w-full h-full object-contain transition-all duration-300 ${
          isModelDark ? 'invert contrast-[1.2]' : 'mix-blend-multiply'
        }`}
        onError={() => setImgError(true)}
      />
    </div>
  )
}
