import { useState, useCallback } from 'react'
import { Loader2, Atom, Network, ArrowRight } from 'lucide-react'
import { DrugSearchInput } from '../components/DrugSearchInput'
import { getDrugNeighbors } from '../api'
import type { DrugNeighbor } from '../api'

// ─── ATC color map ─────────────────────────────────────────────────────────────
const ATC_COLORS: Record<string, string> = {
  A: '#3B82F6',
  B: '#EC4899',
  C: '#10B981',
  J: '#8B5CF6',
  N: '#F59E0B',
  R: '#EF4444',
}
const atcColor = (code?: string | null) =>
  ATC_COLORS[(code ?? '').charAt(0).toUpperCase()] ?? '#6B7280'

// ─── Central drug node ─────────────────────────────────────────────────────────
function CenterNode({ name, onClear }: { name: string; onClear: () => void }) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className="w-20 h-20 rounded-full flex items-center justify-center border-2 border-[var(--accent)] bg-[var(--accent-bg)] cursor-pointer hover:opacity-80 transition-opacity"
        onClick={onClear}
        title="Click to search a different drug"
      >
        <span className="text-[var(--accent)]">
          <Network className="w-6 h-6" />
        </span>
      </div>
      <span className="text-sm font-semibold text-[var(--text-primary)] text-center max-w-[140px] leading-snug">
        {name}
      </span>
      <span className="text-[10px] text-[var(--text-secondary)]">Query drug</span>
    </div>
  )
}

// ─── Neighbor node card ────────────────────────────────────────────────────────
function NeighborCard({
  neighbor,
  rank,
  onClick,
}: {
  neighbor: DrugNeighbor
  rank: number
  onClick: () => void
}) {
  const pct = Math.round(neighbor.similarity * 100)
  const color = atcColor(neighbor.atc_level1)

  return (
    <button
      onClick={onClick}
      className="group flex flex-col gap-3 p-4 rounded bg-[var(--bg-card)] border border-[var(--border-clinical)] hover:border-[var(--accent)] hover:bg-[var(--accent-bg)] transition-all text-left focus:outline-none"
    >
      {/* Rank + Similarity */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-mono text-[var(--text-secondary)]">#{rank}</span>
        <span
          className="text-[10px] font-mono px-2 py-0.5 rounded-full text-white font-semibold"
          style={{ backgroundColor: color }}
        >
          {pct}% similar
        </span>
      </div>

      {/* Colored dot + Name */}
      <div className="flex items-start gap-2">
        <span className="w-2.5 h-2.5 rounded-full shrink-0 mt-0.5" style={{ backgroundColor: color }} />
        <span
          className="text-sm font-medium text-[var(--text-primary)] group-hover:text-[var(--accent)] transition-colors leading-snug"
          style={{ display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}
        >
          {neighbor.name}
        </span>
      </div>

      {/* Category */}
      {(neighbor.categories || neighbor.atc_level1) && (
        <span className="text-[10px] text-[var(--text-secondary)] truncate">
          {neighbor.categories ?? neighbor.atc_level1}
        </span>
      )}

      {/* Molecular weight */}
      {neighbor.molecular_weight && (
        <span className="text-[10px] font-mono text-[var(--text-secondary)]">
          MW: {neighbor.molecular_weight.toFixed(1)} g/mol
        </span>
      )}

      {/* Explore arrow */}
      <div className="flex items-center gap-1 text-[10px] text-[var(--accent)] opacity-0 group-hover:opacity-100 transition-opacity">
        <span>View in Directory</span>
        <ArrowRight className="w-3 h-3" />
      </div>
    </button>
  )
}

// ─── Similarity bar ────────────────────────────────────────────────────────────
function SimilarityBar({ neighbors }: { neighbors: DrugNeighbor[] }) {
  if (neighbors.length === 0) return null

  return (
    <div className="flex flex-col gap-2 p-4 bg-[var(--bg-card)] border border-[var(--border-clinical)] rounded">
      <span className="clinical-label">GNN Similarity Scores</span>
      <div className="flex flex-col gap-2 mt-1">
        {neighbors.map((n, i) => {
          const pct = Math.round(n.similarity * 100)
          const color = atcColor(n.atc_level1)
          return (
            <div key={n.drugbank_id} className="flex items-center gap-3">
              <span className="text-[10px] text-[var(--text-secondary)] font-mono w-3 text-right">{i + 1}</span>
              <span
                className="text-xs text-[var(--text-primary)] truncate"
                style={{ minWidth: 0, flex: '0 0 160px' }}
              >
                {n.name}
              </span>
              <div className="flex-1 bg-[var(--bg-app)] rounded-full h-1.5 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${pct}%`, backgroundColor: color }}
                />
              </div>
              <span className="text-[10px] font-mono text-[var(--text-secondary)] w-8 text-right">{pct}%</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ─── Page ──────────────────────────────────────────────────────────────────────
export function NeighborhoodPage() {
  const [query, setQuery] = useState<string | null>(null)
  const [neighbors, setNeighbors] = useState<DrugNeighbor[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadNeighbors = useCallback(async (name: string) => {
    setQuery(name)
    setNeighbors([])
    setError(null)
    setLoading(true)
    try {
      const res = await getDrugNeighbors(name)
      setNeighbors(res)
    } catch {
      setError('Failed to compute neighbors. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }, [])

  const handleNeighborClick = (name: string) => {
    loadNeighbors(name)
  }

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-normal text-[var(--text-primary)] m-0">Chemical Neighborhood</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-2 mb-0">
          Discover structurally similar drugs using GNN molecular embeddings. Cosine similarity computed on R-GCN learned drug representations.
        </p>
      </div>

      {/* Search */}
      <div className="max-w-[680px]">
        <DrugSearchInput
          id="neighborhood-search"
          placeholder="Search a drug to find its chemical neighbors..."
          onSelect={loadNeighbors}
        />
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex items-center gap-3 py-8 text-[var(--text-secondary)]">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Computing neighbors for {query}...</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="p-4 bg-[var(--bg-card)] border border-[var(--severity-red)] rounded text-sm text-[var(--severity-red)]">
          {error}
        </div>
      )}

      {/* Empty state */}
      {!query && !loading && (
        <div className="flex flex-col items-center justify-center gap-4 py-16 text-center">
          <div className="w-16 h-16 rounded-full bg-[var(--accent-bg)] flex items-center justify-center">
            <Atom className="w-8 h-8 text-[var(--accent)]" />
          </div>
          <div>
            <p className="text-sm font-medium text-[var(--text-primary)] m-0">Search any drug above</p>
            <p className="text-xs text-[var(--text-secondary)] mt-1 m-0">
              We'll show you its 5 most structurally similar drugs from our 12,000+ drug database
            </p>
          </div>
          <div className="grid grid-cols-3 md:grid-cols-5 gap-3 mt-4 max-w-[600px] w-full">
            {['Warfarin', 'Metformin', 'Aspirin', 'Atorvastatin', 'Amiodarone'].map((d) => (
              <button
                key={d}
                onClick={() => loadNeighbors(d)}
                className="py-2 px-3 rounded bg-[var(--bg-card)] border border-[var(--border-clinical)] text-xs text-[var(--text-secondary)] hover:text-[var(--accent)] hover:border-[var(--accent)] transition-all cursor-pointer focus:outline-none bg-transparent"
              >
                {d}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      {!loading && neighbors.length > 0 && query && (
        <div className="flex flex-col gap-6 animate-fade-in">
          {/* Visual: center + neighbors */}
          <div className="flex flex-col gap-4 p-6 bg-[var(--bg-card)] border border-[var(--border-clinical)] rounded">
            {/* Center drug */}
            <div className="flex justify-center">
              <CenterNode name={query} onClear={() => { setQuery(null); setNeighbors([]) }} />
            </div>
            {/* Connector line */}
            <div className="flex justify-center">
              <div className="w-px h-6 bg-[var(--border-clinical)]" />
            </div>
            {/* Neighbor cards grid */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {neighbors.map((n, i) => (
                <NeighborCard
                  key={n.drugbank_id}
                  neighbor={n}
                  rank={i + 1}
                  onClick={() => handleNeighborClick(n.name)}
                />
              ))}
            </div>
          </div>

          {/* Similarity bar chart */}
          <SimilarityBar neighbors={neighbors} />

          {/* Explanation */}
          <div className="p-4 bg-[var(--bg-card)] border border-[var(--border-clinical)] rounded">
            <p className="text-xs text-[var(--text-secondary)] leading-relaxed m-0">
              <strong className="text-[var(--text-primary)]">How this works:</strong> MedSafe's R-GCN model encodes each drug as a 64-dimensional embedding vector that captures structural and pharmacological relationships from graph-based drug interaction data. Chemical neighbors are the drugs with the highest cosine similarity to {query} in this embedding space — meaning the GNN considers them most structurally and pharmacologically related. Clicking a neighbor updates the query.
            </p>
          </div>
        </div>
      )}

      {/* No neighbors found */}
      {!loading && query && neighbors.length === 0 && !error && (
        <div className="flex items-center gap-3 p-4 bg-[var(--bg-card)] border border-[var(--border-clinical)] rounded">
          <Atom className="w-4 h-4 text-[var(--text-secondary)] shrink-0" />
          <span className="text-xs text-[var(--text-secondary)]">
            No neighbors found for <strong>{query}</strong>. GNN embeddings may not include this drug or are not loaded.
          </span>
        </div>
      )}
    </div>
  )
}
