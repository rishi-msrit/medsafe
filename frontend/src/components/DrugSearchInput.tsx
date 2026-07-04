import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Search, X, Loader2 } from 'lucide-react'
import { searchDrugs } from '../api'
import type { DrugSearchResult } from '../api'
import { useDebounce } from '../hooks/useDebounce'

interface DrugSearchInputProps {
  onSelect: (drug: string) => void
  placeholder?: string
  id?: string
  className?: string
  disabled?: boolean
}

export function DrugSearchInput({
  onSelect,
  placeholder = 'Search drug name...',
  id = 'drug-search',
  className = '',
  disabled = false,
}: DrugSearchInputProps) {
  const [query, setQuery] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const debouncedQuery = useDebounce(query, 250)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const { data: results, isLoading } = useQuery({
    queryKey: ['drug-search', debouncedQuery],
    queryFn: () => searchDrugs(debouncedQuery, 8),
    enabled: debouncedQuery.length >= 2,
  })

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleSelect = (drug: DrugSearchResult) => {
    onSelect(drug.name)
    setQuery('')
    setIsOpen(false)
    inputRef.current?.blur()
  }

  return (
    <div ref={wrapperRef} className={`relative ${className}`}>
      <div className="relative">
        {isLoading ? (
          <Loader2 className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-secondary)] animate-spin" />
        ) : (
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-secondary)]" />
        )}
        <input
          ref={inputRef}
          id={id}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
            setIsOpen(true)
          }}
          onFocus={() => query.length >= 2 && setIsOpen(true)}
          placeholder={placeholder}
          disabled={disabled}
          className="clinical-input pl-10 pr-9 bg-[var(--bg-card)] border border-[var(--border-clinical)] text-[var(--text-primary)] rounded w-full py-2.5 px-3 focus:border-[var(--accent)] outline-none"
          autoComplete="off"
          aria-label="Search drug name"
          aria-autocomplete="list"
          aria-controls="drug-suggestions"
          aria-expanded={isOpen}
        />
        {query && (
          <button
            onClick={() => { setQuery(''); setIsOpen(false) }}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors bg-transparent border-0 cursor-pointer p-0"
            aria-label="Clear search"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Autocomplete Suggestions Dropdown */}
      {isOpen && results && results.length > 0 && (
        <div
          id="drug-suggestions"
          role="listbox"
          className="absolute top-full left-0 right-0 mt-1 bg-[var(--bg-card)] border border-[var(--border-clinical)] rounded shadow-sm z-50 overflow-hidden"
        >
          {results.map((drug, i) => (
            <button
              key={drug.name}
              role="option"
              id={`drug-option-${i}`}
              aria-selected={false}
              onClick={() => handleSelect(drug)}
              className="w-full text-left px-4 py-3 flex items-center justify-between hover:bg-[var(--accent-bg)] transition-colors border-b border-[var(--border-clinical)] last:border-0 bg-transparent cursor-pointer"
            >
              <div>
                <p className="text-sm font-medium text-[var(--text-primary)] m-0">{drug.name}</p>
                {drug.categories && (
                  <p className="text-xs text-[var(--text-secondary)] m-0 mt-0.5 truncate max-w-[240px]">
                    {drug.categories.split('|')[0]}
                  </p>
                )}
              </div>
              <div className="text-right shrink-0 flex flex-col items-end gap-1">
                {drug.atc_class && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--accent-bg)] text-[var(--accent)] font-mono">
                    ATC {drug.atc_class}
                  </span>
                )}
                <p className="text-[10px] text-[var(--text-secondary)] m-0">
                  {(drug.match_score * 100).toFixed(0)}% match
                </p>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Empty Search Results State */}
      {isOpen && debouncedQuery.length >= 2 && !isLoading && results?.length === 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 bg-[var(--bg-card)] border border-[var(--border-clinical)] px-4 py-3 z-50 rounded shadow-sm">
          <p className="text-sm text-[var(--text-secondary)] m-0">No drugs found for "{debouncedQuery}"</p>
        </div>
      )}
    </div>
  )
}
