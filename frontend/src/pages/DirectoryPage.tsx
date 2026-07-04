import { useState, useCallback } from 'react'
import { Atom, Loader2, FlaskConical, Weight, Dna, Tag } from 'lucide-react'
import { DrugSearchInput } from '../components/DrugSearchInput'
import { MoleculeViewer } from '../components/MoleculeViewer'
import { HelpBulb } from '../components/InfoBulb'
import { getDrugProfile, getDrugNeighbors } from '../api'
import type { DrugProfile, DrugNeighbor } from '../api'

// ─── ATC color map (matches MolecularExplorerPage) ────────────────────────────
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

// ─── Featured (default) drug list ────────────────────────────────────────────
const FEATURED: { name: string; category: string; atc: string }[] = [
  { name: 'Warfarin',      category: 'Anticoagulant',       atc: 'B' },
  { name: 'Metformin',     category: 'Antidiabetic',         atc: 'A' },
  { name: 'Atorvastatin',  category: 'Statin',               atc: 'C' },
  { name: 'Lisinopril',    category: 'ACE Inhibitor',        atc: 'C' },
  { name: 'Fluoxetine',    category: 'SSRI Antidepressant',  atc: 'N' },
  { name: 'Aspirin',       category: 'NSAID / Antiplatelet', atc: 'B' },
  { name: 'Amiodarone',    category: 'Antiarrhythmic',       atc: 'C' },
  { name: 'Ibuprofen',     category: 'NSAID',                atc: 'M' },
]

// ─── Neighbor pill ────────────────────────────────────────────────────────────
function NeighborPill({
  neighbor,
  onClick,
}: {
  neighbor: DrugNeighbor
  onClick: () => void
}) {
  const pct = Math.round(neighbor.similarity * 100)
  const color = atcColor(neighbor.atc_level1)

  return (
    <button
      onClick={onClick}
      className="flex flex-col items-center justify-between gap-2 p-3 rounded bg-[var(--bg-card)] border border-[var(--border-clinical)] hover:border-[var(--accent)] hover:bg-[var(--accent-bg)] transition-all cursor-pointer group focus:outline-none w-full text-center"
      title={`Load ${neighbor.name}`}
      style={{ minHeight: '100px' }}
    >
      {/* Top row: dot + similarity badge */}
      <div className="flex items-center justify-between w-full">
        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
        <span
          className="text-[10px] font-mono px-1.5 py-0.5 rounded-full text-white font-semibold ml-auto"
          style={{ backgroundColor: color }}
        >
          {pct}%
        </span>
      </div>
      {/* Drug name — clamped to 2 lines */}
      <span
        className="text-xs font-medium text-[var(--text-primary)] group-hover:text-[var(--accent)] transition-colors w-full text-center leading-snug"
        style={{ display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}
      >
        {neighbor.name}
      </span>
      {/* Category — single line, truncated */}
      <span className="text-[10px] text-[var(--text-secondary)] w-full text-center truncate">
        {neighbor.categories ?? neighbor.atc_level1 ?? '—'}
      </span>
    </button>
  )
}


// ─── Main page ────────────────────────────────────────────────────────────────
export function DirectoryPage() {
  const [selectedDrug, setSelectedDrug] = useState<string | null>(null)
  const [profile, setProfile] = useState<DrugProfile | null>(null)
  const [neighbors, setNeighbors] = useState<DrugNeighbor[]>([])
  const [loadingProfile, setLoadingProfile] = useState(false)
  const [loadingNeighbors, setLoadingNeighbors] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadDrug = useCallback(async (name: string) => {
    setSelectedDrug(name)
    setError(null)
    setProfile(null)
    setNeighbors([])
    setLoadingProfile(true)
    setLoadingNeighbors(true)

    try {
      const [profileRes, neighborsRes] = await Promise.allSettled([
        getDrugProfile(name),
        getDrugNeighbors(name),
      ])

      if (profileRes.status === 'fulfilled') {
        setProfile(profileRes.value.drug)
      } else {
        setError('Failed to load drug profile.')
      }

      if (neighborsRes.status === 'fulfilled') {
        setNeighbors(neighborsRes.value)
      }
      // neighbors failure is silent — just shows empty
    } catch (e) {
      setError('An unexpected error occurred.')
    } finally {
      setLoadingProfile(false)
      setLoadingNeighbors(false)
    }
  }, [])

  const mw = profile?.molecular_weight
  const mwDisplay =
    mw !== undefined && mw !== null
      ? typeof mw === 'number'
        ? mw.toFixed(2)
        : String(mw)
      : null

  return (
    <div className="flex flex-col gap-8">
      <HelpBulb
        purpose="Browse and search the full MedSafe drug database. Click any drug to see its molecular structure, chemical metrics, and GNN-computed chemical neighbors."
        inputs="Type a drug name in the search bar, or click any card from the featured list."
        output="Drug profile with 2D structure, metadata, and 5 structurally similar drugs computed from GNN embeddings."
      />

      {/* Page header */}
      <div>
        <h1 className="text-3xl font-normal text-[var(--text-primary)] m-0">Drug Directory</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-2 mb-0">
          {selectedDrug ? `Viewing profile for ${selectedDrug}` : 'Search 12,000+ drugs from DrugBank'}
        </p>
      </div>

      {/* Search bar */}
      <div className="max-w-[680px]">
        <DrugSearchInput
          id="directory-search"
          placeholder="Search any drug — e.g. Warfarin, Aspirin, Metformin..."
          onSelect={loadDrug}
        />
      </div>

      {/* ── Empty state: featured grid ───────────────────────────────────────── */}
      {!selectedDrug && (
        <section className="flex flex-col gap-4">
          <span className="clinical-label">Featured Drugs</span>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {FEATURED.map((drug) => (
              <button
                key={drug.name}
                id={`featured-${drug.name.toLowerCase()}`}
                onClick={() => loadDrug(drug.name)}
                className="clinical-card flex flex-col gap-2 cursor-pointer hover:border-[var(--accent)] hover:bg-[var(--accent-bg)] transition-all text-left focus:outline-none p-4"
              >
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: atcColor(drug.atc) }}
                />
                <span className="text-sm font-medium text-[var(--text-primary)]">{drug.name}</span>
                <span className="text-xs text-[var(--text-secondary)]">{drug.category}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      {/* ── Error state ───────────────────────────────────────────────────────── */}
      {error && (
        <div className="p-4 bg-[var(--bg-card)] border border-[var(--severity-red)] rounded text-sm text-[var(--severity-red)]">
          {error}
        </div>
      )}

      {/* ── Loading skeleton ──────────────────────────────────────────────────── */}
      {loadingProfile && (
        <div className="flex items-center gap-3 py-8 text-[var(--text-secondary)]">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-sm">Loading profile for {selectedDrug}...</span>
        </div>
      )}

      {/* ── Drug profile ──────────────────────────────────────────────────────── */}
      {profile && !loadingProfile && (
        <div className="flex flex-col gap-8 animate-fade-in">
          {/* Main two-column layout */}
          <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
            {/* Left — metadata */}
            <div className="md:col-span-5 flex flex-col gap-5">
              {/* Drug name + ID */}
              <div className="flex flex-col gap-1">
                <h2 className="text-2xl font-normal text-[var(--text-primary)] m-0">{profile.name}</h2>
                {profile.drugbank_id && (
                  <span className="text-xs font-mono text-[var(--text-secondary)]">
                    DrugBank: {profile.drugbank_id}
                  </span>
                )}
              </div>

              {/* Quick metrics */}
              <div className="grid grid-cols-2 gap-3">
                {mwDisplay && (
                  <div className="flex flex-col gap-1 p-3 bg-[var(--bg-card)] border border-[var(--border-clinical)] rounded">
                    <div className="flex items-center gap-1.5 text-[var(--text-secondary)]">
                      <Weight className="w-3.5 h-3.5" />
                      <span className="clinical-label">Mol. Weight</span>
                    </div>
                    <span className="text-sm font-mono font-medium text-[var(--text-primary)]">
                      {mwDisplay} g/mol
                    </span>
                  </div>
                )}
                {profile.atc_level1 && (
                  <div className="flex flex-col gap-1 p-3 bg-[var(--bg-card)] border border-[var(--border-clinical)] rounded">
                    <div className="flex items-center gap-1.5 text-[var(--text-secondary)]">
                      <Tag className="w-3.5 h-3.5" />
                      <span className="clinical-label">ATC Class</span>
                    </div>
                    <span
                      className="text-sm font-mono font-medium"
                      style={{ color: atcColor(profile.atc_level1) }}
                    >
                      {profile.atc_level1}
                    </span>
                  </div>
                )}
                {profile.categories && (
                  <div className="col-span-2 flex flex-col gap-1 p-3 bg-[var(--bg-card)] border border-[var(--border-clinical)] rounded">
                    <div className="flex items-center gap-1.5 text-[var(--text-secondary)]">
                      <FlaskConical className="w-3.5 h-3.5" />
                      <span className="clinical-label">Categories</span>
                    </div>
                    <span className="text-xs text-[var(--text-primary)] leading-relaxed">
                      {profile.categories.split('|').slice(0, 3).join(' · ')}
                    </span>
                  </div>
                )}
              </div>

              {/* Mechanism */}
              {profile.mechanism && (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-1.5 text-[var(--text-secondary)]">
                    <Dna className="w-3.5 h-3.5" />
                    <span className="clinical-label">Mechanism of Action</span>
                  </div>
                  <p className="text-xs text-[var(--text-primary)] leading-relaxed m-0 border-l-2 border-[var(--accent)] pl-3">
                    {profile.mechanism.length > 400
                      ? profile.mechanism.slice(0, 400) + '…'
                      : profile.mechanism}
                  </p>
                </div>
              )}

              {/* Description */}
              {profile.description && (
                <div className="flex flex-col gap-2">
                  <span className="clinical-label">Description</span>
                  <p className="text-xs text-[var(--text-secondary)] leading-relaxed m-0">
                    {profile.description.length > 500
                      ? profile.description.slice(0, 500) + '…'
                      : profile.description}
                  </p>
                </div>
              )}

              {/* SMILES */}
              {profile.smiles && (
                <div className="flex flex-col gap-1">
                  <span className="clinical-label">SMILES</span>
                  <p className="text-[10px] font-mono text-[var(--text-secondary)] break-all leading-relaxed m-0 bg-[var(--bg-app)] p-2 rounded border border-[var(--border-clinical)]">
                    {profile.smiles}
                  </p>
                </div>
              )}
            </div>

            {/* Right — molecule render */}
            <div className="md:col-span-7">
              <MoleculeViewer smiles={profile.smiles} name={profile.name} height="h-[420px]" />
            </div>
          </div>

          {/* ── Chemical Neighborhood ─────────────────────────────────────────── */}
          <section className="flex flex-col gap-4 border-t border-[var(--border-clinical)] pt-6">
            <div className="flex flex-col gap-1">
              <span className="clinical-label">Chemical Neighborhood</span>
              <p className="text-xs text-[var(--text-secondary)] m-0">
                Drugs most structurally similar to {profile.name}, computed via cosine similarity on GNN molecular embeddings.
              </p>
            </div>

            {loadingNeighbors ? (
              <div className="flex items-center gap-2 text-[var(--text-secondary)]">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-xs">Computing neighbors...</span>
              </div>
            ) : neighbors.length > 0 ? (
              <div className="grid grid-cols-5 gap-3">
                {neighbors.map((n) => (
                  <NeighborPill
                    key={n.drugbank_id}
                    neighbor={n}
                    onClick={() => loadDrug(n.name)}
                  />
                ))}
              </div>
            ) : (
              <div className="flex items-center gap-2 p-4 bg-[var(--bg-card)] border border-[var(--border-clinical)] rounded">
                <Atom className="w-4 h-4 text-[var(--text-secondary)]" />
                <span className="text-xs text-[var(--text-secondary)]">
                  Chemical neighborhood unavailable. GNN embeddings may not be loaded.
                </span>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  )
}
