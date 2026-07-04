import { useState, useEffect, useMemo } from 'react'
import { Loader2, Check, Atom } from 'lucide-react'
import { DrugSearchInput } from '../components/DrugSearchInput'
import { HelpBulb } from '../components/InfoBulb'
import { MoleculeViewer } from '../components/MoleculeViewer'
import { tsneData } from '../data/tsneData'
import type { TsnePoint } from '../data/tsneData'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

// --- ATC class colors for t-SNE ---
const ATC_COLORS: Record<string, string> = {
  A: '#3B82F6', // Alimentary - Blue
  B: '#EC4899', // Blood - Pink
  C: '#10B981', // Cardiovascular - Green
  J: '#8B5CF6', // Anti-infectives - Purple
  N: '#F59E0B', // Nervous system - Amber
  R: '#EF4444', // Respiratory - Red
  default: '#6B7280', // Others - Gray
}

const ATC_CLASS_NAMES: Record<string, string> = {
  A: 'Alimentary Tract & Metabolism',
  B: 'Blood & Blood Forming Organs',
  C: 'Cardiovascular System',
  J: 'Anti-infectives',
  N: 'Nervous System',
  R: 'Respiratory System',
  default: 'Other ATC Class',
}

const ATC_CENTERS: Record<string, { x: number; y: number }> = {
  A: { x: 160, y: 150 },
  B: { x: 480, y: 130 },
  C: { x: 650, y: 250 },
  J: { x: 180, y: 380 },
  N: { x: 380, y: 380 },
  R: { x: 450, y: 250 },
  default: { x: 320, y: 200 },
}

interface DrugInfoProps {
  name: string
  drugbank_id?: string
  atc_level1?: string
  atc_class?: string
  atc_codes?: string
  categories?: string
  description?: string
  mechanism?: string
  molecular_weight?: number
  smiles?: string
  groups?: string
}

// MoleculeViewer is now imported from ../components/MoleculeViewer

export function MolecularExplorerPage() {
  const [selectedDrug, setSelectedDrug] = useState<string>('')
  const [drugData, setDrugData] = useState<DrugInfoProps | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  // t-SNE specific state
  const [hoveredPoint, setHoveredPoint] = useState<TsnePoint | null>(null)
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 })
  const [dynamicPoints, setDynamicPoints] = useState<TsnePoint[]>([])

  // Combine static and dynamic points
  const points = useMemo(() => {
    return [...tsneData, ...dynamicPoints]
  }, [dynamicPoints])

  // Scaled coordinates mapping to spread them out in the 800x550 SVG viewBox
  const mappedPoints = useMemo(() => {
    if (points.length === 0) return []
    const xs = points.map(p => p.x)
    const ys = points.map(p => p.y)
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)

    const spanX = maxX - minX || 1
    const spanY = maxY - minY || 1

    return points.map(p => ({
      ...p,
      // Map x to range [60, 740]
      x: 60 + ((p.x - minX) / spanX) * 680,
      // Map y to range [60, 490]
      y: 60 + ((p.y - minY) / spanY) * 430,
    }))
  }, [points])

  // Locate selected drug in points list
  const selectedPoint = useMemo(() => {
    if (!selectedDrug) return null
    return mappedPoints.find(p => p.name.toLowerCase() === selectedDrug.toLowerCase()) || null
  }, [selectedDrug, mappedPoints])

  // Compute 5 nearest neighbors in coordinate space
  const nearestNeighbors = useMemo(() => {
    if (!selectedPoint) return []
    return mappedPoints
      .filter(p => p.name !== selectedPoint.name)
      .map(p => {
        const dist = Math.sqrt(
          Math.pow(p.x - selectedPoint.x, 2) + Math.pow(p.y - selectedPoint.y, 2)
        )
        return { point: p, dist }
      })
      .sort((a, b) => a.dist - b.dist)
      .slice(0, 5)
      .map(n => n.point)
  }, [selectedPoint, mappedPoints])

  // Copy SMILES logic
  const handleCopySmiles = (smiles: string) => {
    navigator.clipboard.writeText(smiles)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Fetch drug details when selectedDrug changes
  useEffect(() => {
    if (!selectedDrug) {
      setDrugData(null)
      setError(null)
      return
    }

    setLoading(true)
    setError(null)

    // Check if the selected drug needs to be added to dynamic coordinates
    const exists = points.some(p => p.name.toLowerCase() === selectedDrug.toLowerCase())

    fetch(`${API}/drugs/${encodeURIComponent(selectedDrug)}`)
      .then(r => r.json())
      .then(data => {
        if (data.found && data.drug) {
          setDrugData(data.drug as DrugInfoProps)
          
          // Place dynamically in t-SNE space if it was not already in the dataset
          if (!exists) {
            const atcClass = data.drug.atc_class || 'default'
            const center = ATC_CENTERS[atcClass] || ATC_CENTERS.default
            const randOffset = () => (Math.random() - 0.5) * 30
            const newPoint: TsnePoint = {
              name: data.drug.name,
              x: center.x + randOffset(),
              y: center.y + randOffset(),
              atcClass: atcClass,
              atcClassName: ATC_CLASS_NAMES[atcClass] || ATC_CLASS_NAMES.default,
            }
            setDynamicPoints(prev => [...prev, newPoint])
          }
        } else {
          // Fallback if detail not found
          setDrugData({ name: selectedDrug, drugbank_id: data.drug?.drugbank_id })
          setError('Detailed chemical records not available for this drug.')
        }
      })
      .catch(() => {
        setDrugData({ name: selectedDrug })
        setError('Could not load drug details. Verify API connectivity.')
      })
      .finally(() => setLoading(false))
  }, [selectedDrug])

  // Handle t-SNE point hover
  const handleMouseMove = (e: React.MouseEvent, point: TsnePoint) => {
    const rect = e.currentTarget.getBoundingClientRect()
    setTooltipPos({
      x: e.clientX - rect.left + 15,
      y: e.clientY - rect.top + 15,
    })
    setHoveredPoint(point)
  }

  return (
    <div className="flex flex-col gap-8 relative w-full">
      <HelpBulb
        purpose="Search a drug to view its chemical details and structures. Observe the coordinate mapping in the t-SNE Molecular Space scatter plot below."
        inputs="Type a drug name in the search box to load structure details. Hover or click coordinates in the plot to highlight nearest neighbors."
        output="A 2D structure drawing, categories list, clinical mechanism of action, and t-SNE plot highlighting drug clusters and neighboring molecules."
      />

      {/* Heading - DM Serif Display */}
      <div>
        <h1 className="text-3xl font-normal text-[var(--text-primary)] m-0">
          Molecular Space
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1.5 m-0 font-normal">
          Each point is a drug. Nearby drugs share similar molecular structure. Colored by drug class.
        </p>
      </div>

      {/* --- Section 1: Chemical Structure Search & Info --- */}
      <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col gap-4 max-w-[760px] w-full mx-auto">
        <label className="clinical-label block">Search Medication</label>
        <DrugSearchInput
          id="molecular-drug-search"
          onSelect={setSelectedDrug}
          placeholder="Enter drug name to explore..."
        />
        {selectedDrug && (
          <div className="flex items-center justify-between mt-1.5">
            <span className="text-xs text-[var(--text-secondary)] font-normal">
              Exploring: <strong className="text-[var(--text-primary)] font-mono">{selectedDrug}</strong>
            </span>
            <button
              onClick={() => { setSelectedDrug(''); setDrugData(null) }}
              className="text-xs text-[var(--accent)] hover:underline font-medium bg-transparent border-0 cursor-pointer p-0"
            >
              Clear Search
            </button>
          </div>
        )}
      </div>

      {/* Loading */}
      {loading && (
        <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] py-8 flex items-center justify-center gap-3 text-[var(--text-secondary)] max-w-[760px] w-full mx-auto">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span className="text-xs font-medium">Loading structure records...</span>
        </div>
      )}

      {/* Error Banner */}
      {error && !loading && (
        <div className="p-3 bg-[var(--bg-card)] border border-yellow-500/20 text-[var(--severity-amber)] text-xs rounded max-w-[760px] w-full mx-auto">
          {error}
        </div>
      )}

      {/* Structure details layout */}
      {!loading && drugData && (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-start animate-fade-in">
          {/* Left: Structure details */}
          <div className="md:col-span-6 flex flex-col gap-4">
            <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col items-center text-center gap-4 bg-[var(--bg-card)]">
              <MoleculeViewer smiles={drugData.smiles} name={drugData.name} />
              <div>
                <h2 className="text-lg font-semibold text-[var(--text-primary)] m-0 font-mono">{drugData.name}</h2>
                {drugData.drugbank_id && (
                  <p className="text-[10px] font-mono text-[var(--text-secondary)] mt-1 m-0">{drugData.drugbank_id}</p>
                )}
              </div>

              {drugData.atc_level1 && (
                <div className="px-2.5 py-1 rounded bg-[var(--accent-bg)] text-[var(--accent)] text-[10px] font-mono border border-[var(--border-clinical)]">
                  ATC: {drugData.atc_level1}
                </div>
              )}

              <div className="flex gap-3 w-full pt-1">
                {drugData.drugbank_id && (
                  <a
                    href={`https://go.drugbank.com/drugs/${drugData.drugbank_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    id={`drugbank-link-${drugData.drugbank_id}`}
                    className="btn-secondary text-xs flex-1 py-1.5"
                  >
                    DrugBank
                  </a>
                )}
                <a
                  href={`https://pubchem.ncbi.nlm.nih.gov/compound/${encodeURIComponent(drugData.name)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  id={`pubchem-link-${drugData.name}`}
                  className="btn-secondary text-xs flex-1 py-1.5"
                >
                  PubChem
                </a>
              </div>
            </div>
          </div>

          {/* Right: Text properties */}
          <div className="md:col-span-6 flex flex-col gap-6">
            <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col gap-4 bg-[var(--bg-card)]">
              <div>
                <span className="clinical-label block mb-2">Pharmacological Groups</span>
                {drugData.groups ? (
                  <div className="flex flex-wrap gap-1.5">
                    {drugData.groups.split(' ').map(g => (
                      <span key={g} className="text-[11px] px-2.5 py-1 rounded bg-[var(--bg-app)] text-[var(--text-primary)] border border-[var(--border-clinical)] font-mono">
                        {g.trim()}
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-[var(--text-secondary)] m-0 font-normal">No pharmacological classifications recorded.</p>
                )}
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {drugData.molecular_weight && (
                  <div className="clinical-card p-3 border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col">
                    <span className="clinical-label">Molecular Weight</span>
                    <span className="text-sm font-semibold text-[var(--text-primary)] font-mono mt-0.5">
                      {Number(drugData.molecular_weight).toFixed(2)} g/mol
                    </span>
                  </div>
                )}
                {drugData.atc_class && (
                  <div className="clinical-card p-3 border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col">
                    <span className="clinical-label">ATC Class Code</span>
                    <span className="text-sm font-semibold text-[var(--text-primary)] font-mono mt-0.5">
                      {drugData.atc_class}
                    </span>
                  </div>
                )}
              </div>

              {drugData.smiles && (
                <div className="flex flex-col gap-2">
                  <span className="clinical-label block">SMILES string</span>
                  <div className="flex items-center gap-2 bg-[var(--bg-app)] rounded p-2.5 border border-[var(--border-clinical)]">
                    <code className="text-xs text-[var(--text-primary)] break-all flex-1 font-mono leading-normal">{drugData.smiles}</code>
                    <button
                      onClick={() => handleCopySmiles(drugData.smiles!)}
                      className="shrink-0 p-1.5 rounded hover:bg-[var(--border-clinical)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors bg-transparent border-0 cursor-pointer"
                      title="Copy SMILES string"
                    >
                      {copied ? <Check className="w-4 h-4 text-green-600" /> : <Atom className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {drugData.description && (
              <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col gap-2 bg-[var(--bg-card)]">
                <span className="clinical-label block">Clinical Description</span>
                <p className="text-sm text-[var(--text-primary)] leading-relaxed m-0 font-normal">{drugData.description}</p>
              </div>
            )}

            {drugData.mechanism && (
              <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col gap-2 bg-[var(--bg-card)]">
                <span className="clinical-label block">Mechanism of Action</span>
                <p className="text-sm text-[var(--text-primary)] leading-relaxed m-0 font-normal">{drugData.mechanism}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!selectedDrug && (
        <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] text-center py-8 bg-[var(--bg-card)] max-w-[760px] w-full mx-auto">
          <h3 className="text-base font-semibold text-[var(--text-primary)] m-0">Search a Drug</h3>
          <p className="text-xs text-[var(--text-secondary)] mt-2 leading-relaxed max-w-md mx-auto m-0 font-normal">
            Type any drug name in the search bar above to view its 2D chemical structure, description, and properties.
          </p>
        </div>
      )}

      {/* --- Section 2: Molecular Space Representation (t-SNE SVG Plot) --- */}
      <div className="mt-8 border-t border-[#E2DED8] dark:border-[#2C2C2E] pt-8 flex flex-col gap-4">
        <div>
          <h2 className="text-sm font-semibold text-[var(--text-primary)] m-0 uppercase tracking-wider">
            Molecular Representation Space
          </h2>
          <p className="text-xs text-[var(--text-secondary)] mt-1.5 m-0 font-normal">
            Click any dot in the 2D t-SNE coordinates mapping space below to inspect its chemical properties at the top of the page.
          </p>
        </div>

        {/* SVG Plot Container - direct on background, no card wrap */}
        <div className="relative w-full overflow-hidden p-2 flex justify-center bg-white dark:bg-[#1C1C1E] rounded border border-[#E2DED8] dark:border-[#2C2C2E]">
          <svg
            viewBox="0 0 800 550"
            className="w-full h-auto max-h-[60vh] select-none"
            id="tsne-svg-plot"
          >
            {/* Neighbor lines */}
            {selectedPoint &&
              nearestNeighbors.map(neighbor => (
                <line
                  key={neighbor.name}
                  x1={selectedPoint.x}
                  y1={selectedPoint.y}
                  x2={neighbor.x}
                  y2={neighbor.y}
                  stroke="var(--text-secondary)"
                  strokeWidth="1"
                  strokeDasharray="3,3"
                  className="opacity-50"
                />
              ))}

            {/* Scatter dots */}
            {mappedPoints.map(point => {
              const isSelected = selectedPoint && selectedPoint.name === point.name
              const color = ATC_COLORS[point.atcClass] || ATC_COLORS.default
              const isNeighbor = nearestNeighbors.some(n => n.name === point.name)

              return (
                <g
                  key={point.name}
                  onMouseMove={e => handleMouseMove(e, point)}
                  onMouseLeave={() => setHoveredPoint(null)}
                  onClick={() => setSelectedDrug(point.name)}
                  className="cursor-pointer"
                >
                  {isSelected && (
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r="12"
                      fill="none"
                      stroke="var(--text-primary)"
                      strokeWidth="1.5"
                    />
                  )}
                  {isNeighbor && (
                    <circle
                      cx={point.x}
                      cy={point.y}
                      r="8"
                      fill="none"
                      stroke={color}
                      strokeWidth="1"
                      strokeDasharray="2,2"
                    />
                  )}
                  <circle
                    cx={point.x}
                    cy={point.y}
                    r={isSelected ? 6 : isNeighbor ? 5 : 4}
                    fill={color}
                  />
                </g>
              )
            })}

            {/* Selection Text label */}
            {selectedPoint && (
              <text
                x={selectedPoint.x}
                y={selectedPoint.y - 18}
                textAnchor="middle"
                className="text-[10px] font-mono font-semibold fill-[var(--text-primary)]"
              >
                {selectedPoint.name}
              </text>
            )}
          </svg>

          {/* Tooltip */}
          {hoveredPoint && (
            <div
              className="absolute z-40 tooltip-custom select-none leading-normal pointer-events-none"
              style={{
                left: `${tooltipPos.x}px`,
                top: `${tooltipPos.y}px`,
              }}
            >
              <p className="font-semibold m-0 font-mono">{hoveredPoint.name}</p>
              <p className="text-[10px] text-gray-400 m-0 mt-0.5">
                Class: {hoveredPoint.atcClassName} (ATC-{hoveredPoint.atcClass})
              </p>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 py-2">
          {Object.entries(ATC_COLORS).map(([key, color]) => (
            <div key={key} className="flex items-center gap-2">
              <span
                className="w-3.5 h-3.5 rounded-full shrink-0"
                style={{ backgroundColor: color }}
              />
              <span className="text-xs text-[var(--text-secondary)] font-mono">
                {ATC_CLASS_NAMES[key] || ATC_CLASS_NAMES.default}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
