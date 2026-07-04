import { useState, useCallback } from 'react'
import { useEffect } from 'react'

export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay)
    return () => clearTimeout(timer)
  }, [value, delay])

  return debouncedValue
}

export function useDrugList(maxDrugs = 15) {
  const [drugs, setDrugs] = useState<string[]>([])

  const addDrug = useCallback((drug: string) => {
    const cleaned = drug.trim()
    if (!cleaned) return
    if (drugs.length >= maxDrugs) return
    if (drugs.some(d => d.toLowerCase() === cleaned.toLowerCase())) return
    setDrugs(prev => [...prev, cleaned])
  }, [drugs, maxDrugs])

  const removeDrug = useCallback((drug: string) => {
    setDrugs(prev => prev.filter(d => d !== drug))
  }, [])

  const clearDrugs = useCallback(() => setDrugs([]), [])

  return { drugs, addDrug, removeDrug, clearDrugs }
}
