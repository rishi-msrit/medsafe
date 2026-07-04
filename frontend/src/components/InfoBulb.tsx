import { useState, useRef, useEffect } from 'react'
import { Lightbulb } from 'lucide-react'

interface HelpBulbProps {
  purpose: string
  inputs: string
  output: string
  id?: string
}

export function HelpBulb({ purpose, inputs, output, id = 'help-bulb' }: HelpBulbProps) {
  const [isOpen, setIsOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Auto-close on click outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  return (
    <div ref={containerRef} className="absolute right-0 top-0 z-30" id={id}>
      {/* Help Bulb Button - 32px circle, background #F0EDE8, border 1px solid #D1CFC9 */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="help-bulb-btn focus:outline-none"
        aria-label="Help and clinical usage instructions"
        aria-expanded={isOpen}
      >
        <Lightbulb className="w-4 h-4 stroke-[1.5]" />
      </button>

      {/* Clinical Instruction Card Dropdown - Border #E2DED8, Plain Text, No Colors, No Icons */}
      {isOpen && (
        <div className="absolute right-0 top-10 w-72 md:w-80 bg-[var(--bg-card)] border border-[#E2DED8] dark:border-[#2C2C2E] p-5 rounded animate-fade-in shadow-none text-left">
          <div className="flex flex-col gap-4">
            <div>
              <span className="text-[10px] font-semibold text-[var(--text-secondary)] uppercase tracking-wider block mb-1">
                What this page does
              </span>
              <p className="text-xs text-[var(--text-primary)] leading-normal m-0 font-normal">
                {purpose}
              </p>
            </div>

            <div>
              <span className="text-[10px] font-semibold text-[var(--text-secondary)] uppercase tracking-wider block mb-1">
                What you should enter
              </span>
              <p className="text-xs text-[var(--text-primary)] leading-normal m-0 font-normal">
                {inputs}
              </p>
            </div>

            <div>
              <span className="text-[10px] font-semibold text-[var(--text-secondary)] uppercase tracking-wider block mb-1">
                Expected output
              </span>
              <p className="text-xs text-[var(--text-primary)] leading-normal m-0 font-normal">
                {output}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
