import { useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Plus, X, Loader2, ChevronDown } from 'lucide-react'
import { analyzePolypharmacy, getAlternatives } from '../api'
import type { SafetyReport, InteractionCard } from '../api'
import { DrugSearchInput } from '../components/DrugSearchInput'
import { useDrugList } from '../hooks/useDebounce'
import { HelpBulb } from '../components/InfoBulb'

const SEVERITY_CONFIG = {
  minor:           { borderClass: 'border-l-[3px] border-l-gray-400',       textClass: 'text-gray-500 dark:text-gray-400', label: 'Minor' },
  moderate:        { borderClass: 'border-l-[3px] border-l-[var(--severity-yellow)]', textClass: 'text-[var(--severity-yellow)]', label: 'Moderate' },
  major:           { borderClass: 'border-l-[3px] border-l-[var(--severity-amber)]',  textClass: 'text-[var(--severity-amber)]',  label: 'Major' },
  contraindicated: { borderClass: 'border-l-[3px] border-l-[var(--severity-red)]',    textClass: 'text-[var(--severity-red)]',    label: 'Contraindicated' },
}

const DEMO_DRUGS = ['Warfarin', 'Aspirin', 'Ibuprofen', 'Metformin', 'Lisinopril']

// Helper to strip low-data warnings from clinical implications
const cleanClinicalImplication = (text: string) => {
  if (!text) return ''
  return text.replace(/⚠️\s*Limited data \(\d+ source\(s\)\)\. Interaction status uncertain\. Consult a pharmacist for personalized advice\.\s*/gi, '')
}

function InteractionCardComponent({
  interaction,
  expanded,
  onToggle
}: {
  interaction: InteractionCard
  expanded: boolean
  onToggle: () => void
}) {
  const severityKey = interaction.severity_label as keyof typeof SEVERITY_CONFIG
  const cfg = SEVERITY_CONFIG[severityKey] || SEVERITY_CONFIG.minor

  return (
    <div
      className={`clinical-card flex flex-col p-4 mb-3 border border-[#E2DED8] dark:border-[#2C2C2E] bg-[var(--bg-card)] cursor-pointer hover:bg-[var(--bg-app)] transition-colors select-none ${cfg.borderClass}`}
      id={`interaction-${interaction.drug_a}-${interaction.drug_b}`.replace(/\s+/g, '-').toLowerCase()}
      onClick={onToggle}
    >
      <div className="w-full flex items-center justify-between p-0">
        <div className="flex-1 min-w-0 pr-4">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="font-semibold text-sm text-[var(--text-primary)] font-mono">{interaction.drug_a}</span>
            <span className="text-[var(--text-secondary)] text-xs font-mono">+</span>
            <span className="font-semibold text-sm text-[var(--text-primary)] font-mono">{interaction.drug_b}</span>
          </div>
          {!expanded && (
            <p className="text-xs text-[var(--text-secondary)] mt-1.5 truncate m-0 font-normal">
              {interaction.plain_english.substring(0, 100)}...
            </p>
          )}
        </div>
        <div className="flex items-center gap-4 shrink-0">
          {/* Text-only outline badge in severity color */}
          <span 
            className="sev-badge"
            style={{
              borderColor: 'currentColor',
              color: `var(--severity-${interaction.severity_label === 'contraindicated' ? 'red' : interaction.severity_label === 'minor' ? 'secondary' : interaction.severity_label})`
            }}
          >
            {cfg.label}
          </span>
          <span className="text-xs text-[var(--text-secondary)] min-w-[32px] text-right font-mono">
            {(interaction.confidence * 100).toFixed(0)}%
          </span>
          <ChevronDown 
            className={`w-4 h-4 text-[var(--text-secondary)] transition-transform duration-200 ${
              expanded ? 'rotate-180' : ''
            }`} 
          />
        </div>
      </div>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-[var(--border-clinical)] animate-fade-in flex flex-col gap-3">
          <div className="bg-[var(--bg-app)] rounded p-4 border border-[var(--border-clinical)]">
            <p className="text-sm text-[var(--text-primary)] leading-relaxed m-0 font-normal">
              {interaction.plain_english}
            </p>
          </div>
          {interaction.clinical_implication && (
            <div className="p-3 bg-[var(--bg-app)] border-l-2 border-l-[var(--accent)] text-xs text-[var(--text-primary)] leading-normal font-normal">
              Clinical Implication: {cleanClinicalImplication(interaction.clinical_implication)}
            </div>
          )}
          <div className="grid grid-cols-3 gap-3 w-full">
            <div className="clinical-card p-3 flex flex-col gap-0.5 border-[#E2DED8] dark:border-[#2C2C2E]">
              <span className="clinical-label">Confidence</span>
              <span className="text-sm font-semibold text-[var(--text-primary)] font-mono">
                {(interaction.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="clinical-card p-3 flex flex-col gap-0.5 border-[#E2DED8] dark:border-[#2C2C2E]">
              <span className="clinical-label">Data Records</span>
              <span className="text-sm font-semibold text-[var(--text-primary)] font-mono">
                {interaction.support_count}
              </span>
            </div>
            <div className="clinical-card p-3 flex flex-col gap-0.5 border-[#E2DED8] dark:border-[#2C2C2E]">
              <span className="clinical-label">CYP Enzymes</span>
              <span className="text-xs text-[var(--text-primary)] font-mono truncate">
                {interaction.cyp_enzymes.length > 0 ? interaction.cyp_enzymes.join(', ') : 'None'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function InteractionMatrix({ report }: { report: SafetyReport }) {
  const drugs = report.drug_list
  const matrix = report.interaction_matrix

  const getMatrixCellBg = (sev: number) => {
    return [
      'bg-[var(--border-clinical)] opacity-30',
      'bg-[var(--text-secondary)] opacity-40',
      'bg-[var(--severity-yellow)]',
      'bg-[var(--severity-amber)]',
      'bg-[var(--severity-red)]',
    ][sev] ?? 'bg-[var(--border-clinical)] opacity-30'
  }

  const getSeverityName = (sev: number) => {
    return ['No interaction', 'Minor interaction', 'Moderate interaction', 'Major interaction', 'Contraindicated'][sev] || 'Unknown'
  }

  return (
    <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E]">
      <h3 className="clinical-label mb-4 mt-0">
        Interaction Matrix
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-medium border-collapse">
          <thead>
            <tr>
              <th className="w-[100px] min-w-[100px]" />
              {drugs.map(d => (
                <th
                  key={d}
                  className="px-1 py-1.5 text-[var(--text-secondary)] font-medium text-center truncate max-w-[64px] text-[10px] font-mono"
                  title={d}
                >
                  {d.substring(0, 8)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {drugs.map(drugA => (
              <tr key={drugA} className="border-b border-[#E2DED8] dark:border-[#2C2C2E] last:border-0">
                <td
                  className="text-[var(--text-primary)] py-2 pr-2 font-medium truncate max-w-[100px] text-[11px] font-mono"
                  title={drugA}
                >
                  {drugA.substring(0, 12)}
                </td>
                {drugs.map(drugB => {
                  if (drugA === drugB) {
                    return (
                      <td key={drugB} className="p-0.5">
                        <div className="w-7 h-7 mx-auto rounded bg-[var(--bg-app)] border border-[var(--border-clinical)] opacity-40" />
                      </td>
                    )
                  }
                  const sev = matrix[drugA]?.[drugB] ?? 0
                  return (
                    <td key={drugB} className="p-0.5">
                      <div
                        className={`w-7 h-7 mx-auto rounded cursor-pointer ${getMatrixCellBg(sev)}`}
                        title={`${drugA} + ${drugB}: ${getSeverityName(sev)}`}
                        role="img"
                        aria-label={`${drugA} and ${drugB}: ${getSeverityName(sev)}`}
                      />
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center gap-4 mt-4 justify-end flex-wrap">
        {[
          ['None', 'bg-[var(--border-clinical)] opacity-30'],
          ['Minor', 'bg-[var(--text-secondary)] opacity-40'],
          ['Moderate', 'bg-[var(--severity-yellow)]'],
          ['Major', 'bg-[var(--severity-amber)]'],
          ['Contraindicated', 'bg-[var(--severity-red)]']
        ].map(([label, bgClass]) => (
          <div key={label} className="flex items-center gap-2">
            <div className={`w-3.5 h-3.5 rounded ${bgClass}`} />
            <span className="text-[10px] text-[var(--text-secondary)] font-medium">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function SaferAlternativesSection({
  drugToReplace,
  otherDrugs
}: {
  drugToReplace: string
  otherDrugs: string[]
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['safer-alternatives-patient-safety', drugToReplace, otherDrugs],
    queryFn: () => getAlternatives(drugToReplace, otherDrugs),
    enabled: !!drugToReplace && otherDrugs.length > 0,
  })

  if (isLoading) {
    return (
      <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] py-6 flex items-center justify-center gap-3 text-[var(--text-secondary)]">
        <Loader2 className="w-4 h-4 animate-spin" />
        <span className="text-xs">Finding safer alternatives for {drugToReplace}...</span>
      </div>
    )
  }

  if (error || !data || !data.alternatives || data.alternatives.length === 0) {
    return null
  }

  // Show only up to 3 alternatives
  const items = data.alternatives.slice(0, 3)

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-sm font-semibold text-[var(--text-primary)] m-0 font-mono">
        Safer Alternatives for {drugToReplace}
      </h3>
      <p className="text-xs text-[var(--text-secondary)] m-0">
        The GNN identified these structurally similar medicines that carry lower interaction risks with your other drugs.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-1">
        {items.map((alt) => (
          <div
            key={alt.drug_name}
            className="clinical-card border border-[#E2DED8] dark:border-[#2C2C2E] border-l-4 border-l-[var(--severity-green)] p-4 flex flex-col justify-between bg-[var(--bg-card)]"
          >
            <div>
              <h4 className="text-sm font-medium text-[var(--text-primary)] m-0 font-mono font-semibold">
                {alt.drug_name}
              </h4>
              <p className="text-xs text-[var(--text-secondary)] mt-1.5 leading-normal">
                {alt.mechanism_explanation}
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-[var(--border-clinical)] flex items-center justify-between text-[11px] text-[var(--text-secondary)]">
              <span>Similarity: {(alt.similarity_score * 100).toFixed(0)}%</span>
              {alt.risk_reduction_pct > 0 ? (
                <span className="font-semibold text-[var(--severity-green)]">
                  Risk reduced by {Math.abs(alt.risk_reduction_pct).toFixed(0)}%
                </span>
              ) : (
                <span className="font-semibold text-[var(--severity-green)] font-sans">
                  Safe alternative (No interaction)
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export function PatientAnalyzerPage() {
  const { drugs, addDrug, removeDrug, clearDrugs } = useDrugList(15)
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set())

  const { mutate: analyze, data: report, isPending, error, reset } = useMutation({
    mutationFn: analyzePolypharmacy,
  })

  const toggleCard = (key: string) => setExpandedCards(prev => {
    const next = new Set(prev)
    next.has(key) ? next.delete(key) : next.add(key)
    return next
  })

  const handleAnalyze = () => {
    if (drugs.length >= 2) analyze(drugs)
  }

  const loadDemo = () => {
    clearDrugs()
    DEMO_DRUGS.forEach(d => addDrug(d))
  }

  const getRiskColorClass = (score: number) => {
    if (score > 70) return 'text-[var(--severity-red)]'
    if (score > 35) return 'text-[var(--severity-amber)]'
    return 'text-[var(--severity-green)]'
  }
  
  const getRiskColorHex = (score: number) => {
    if (score > 70) return 'var(--severity-red)'
    if (score > 35) return 'var(--severity-amber)'
    return 'var(--severity-green)'
  }

  // Determine other drugs to pass to safer alternatives section
  const otherDrugs = report?.risk_culprit 
    ? drugs.filter(d => d.toLowerCase() !== report.risk_culprit?.toLowerCase()) 
    : []

  // Check if high-risk interaction exists to show alternatives section
  const hasHighRisk = report 
    ? (report.overall_risk_score > 50 || report.flagged_interactions.some(i => i.severity_label === 'major' || i.severity_label === 'contraindicated'))
    : false

  return (
    <div className="flex flex-col gap-8 relative">
      <HelpBulb
        purpose="Analyze your complete multi-drug medication list to compute an overall safety risk score and check for dangerous drug-drug interactions."
        inputs="Type drug names in the search box to add them to your medication list. You can load a sample list with 'Load Demo'. Click the Analyze button when ready."
        output="A total risk score (0-100) with a safety explanation, a visual matrix grid mapping conflicts, individual interaction detail cards, risk contribution analysis, and recommended safer alternatives for the highest-risk drug."
      />

      {/* Header - DM Serif Display */}
      <div>
        <h1 className="text-3xl font-normal text-[var(--text-primary)] m-0">
          Medication Safety Check
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1.5 m-0 font-normal">
          Enter all medications you are currently taking to check for interactions.
        </p>
      </div>

      {/* Main Grid Layout: Form on Left, Context on Right (when no report) */}
      {!report ? (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
          {/* Left: Input Regimen Form */}
          <div className="md:col-span-7 flex flex-col gap-4">
            <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <h2 className="clinical-label m-0">
                  Medication Regimen
                </h2>
                <div className="flex items-center gap-4">
                  <span className="text-xs text-[var(--text-secondary)] font-mono">{drugs.length} / 15</span>
                  <button
                    onClick={loadDemo}
                    className="bg-transparent border-0 text-xs font-medium text-[var(--accent)] hover:underline cursor-pointer p-0"
                  >
                    Load Demo
                  </button>
                  {drugs.length > 0 && (
                    <button
                      onClick={() => { clearDrugs(); reset() }}
                      className="bg-transparent border-0 text-xs font-medium text-red-500 hover:underline cursor-pointer p-0"
                    >
                      Clear All
                    </button>
                  )}
                </div>
              </div>

              {/* Selected Drugs list - Prescription chips style */}
              {drugs.length > 0 && (
                <div className="flex flex-wrap gap-2 py-1">
                  {drugs.map(drug => (
                    <div
                      key={drug}
                      className="prescription-tag flex items-center gap-1.5 text-xs text-[var(--text-primary)] px-2.5 py-1 rounded font-mono"
                    >
                      <span>{drug}</span>
                      <button
                        onClick={() => removeDrug(drug)}
                        className="bg-transparent border-0 cursor-pointer flex p-0 text-gray-500 hover:text-red-500 transition-colors"
                        aria-label={`Remove ${drug}`}
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <DrugSearchInput
                id="drug-search-main"
                onSelect={addDrug}
                placeholder="Type a drug name..."
                disabled={drugs.length >= 15}
              />

              {/* Manual Addition Trigger */}
              <div className="flex gap-2">
                <input
                  id="drug-input-manual"
                  type="text"
                  placeholder="Or type exact name..."
                  className="clinical-input flex-1"
                  onKeyDown={e => {
                    if (e.key === 'Enter') {
                      const val = (e.target as HTMLInputElement).value.trim()
                      if (val) { addDrug(val); (e.target as HTMLInputElement).value = '' }
                    }
                  }}
                />
                <button
                  onClick={() => {
                    const inp = document.getElementById('drug-input-manual') as HTMLInputElement
                    if (inp?.value.trim()) { addDrug(inp.value.trim()); inp.value = '' }
                  }}
                  className="btn-secondary flex items-center gap-1.5"
                >
                  <Plus className="w-4 h-4" /> Add
                </button>
              </div>

              <button
                id="analyze-btn"
                onClick={handleAnalyze}
                disabled={drugs.length < 2 || isPending}
                className="btn-primary w-full py-3 mt-2"
              >
                {isPending ? (
                  <span className="flex items-center justify-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Analyzing medications...
                  </span>
                ) : (
                  drugs.length > 0 ? `Analyze ${drugs.length} Medications` : 'Analyze Medications'
                )}
              </button>
            </div>

            {/* Error Output */}
            {error && (
              <div className="clinical-card border-red-500 flex flex-col gap-1.5">
                <p className="font-semibold text-sm text-[var(--severity-red)] m-0 font-sans">Analysis Failed</p>
                <p className="text-xs text-[var(--text-secondary)] m-0 font-normal">
                  {(error as Error).message || 'Could not connect to the API. Make sure the backend server is active.'}
                </p>
              </div>
            )}
          </div>

          {/* Right: Clinical Context Sidebar */}
          <div className="md:col-span-5 flex flex-col gap-4">
            <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] p-6 bg-[var(--bg-card)]">
              <h3 className="clinical-label m-0 mb-3">Clinical Context</h3>
              <p className="text-xs text-[var(--text-secondary)] leading-relaxed m-0 font-normal">
                Multi-drug therapy (polypharmacy) carries high risks of unknown drug-drug interactions. Traditional databases check only cataloged drug pairs, whereas MedSafe utilizes a Graph Neural Network (GNN) to learn relational feature representations and predict adverse reactions based on molecular structural graphs.
              </p>
              <div className="mt-4 pt-4 border-t border-[var(--border-clinical)]">
                <span className="clinical-label block mb-2">Instructions</span>
                <ul className="text-xs text-[var(--text-secondary)] pl-4 space-y-1.5 m-0 font-normal list-disc">
                  <li>Search or add drugs to compile your current medication list.</li>
                  <li>Use the 'Load Demo' link to populate a sample clinical list.</li>
                  <li>Run analysis to calculate the cumulative risk score.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      ) : (
        /* Dashboard Layout: When results are loaded */
        <div className="flex flex-col gap-8 animate-fade-in">
          {/* Top layout: inputs section remains compact so user can modify list */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 bg-[var(--bg-card)] border border-[#E2DED8] dark:border-[#2C2C2E] rounded">
            <div className="flex flex-col gap-1 pr-4">
              <span className="clinical-label">Active Regimen ({drugs.length} Drugs)</span>
              <p className="text-xs text-[var(--text-secondary)] font-mono m-0 truncate max-w-lg">
                {drugs.join(', ')}
              </p>
            </div>
            <button
              onClick={() => reset()}
              className="btn-secondary text-xs py-1.5 shrink-0"
            >
              Modify List
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
            {/* Left Column: Overall Risk & Contribution Analysis (col-span-5) */}
            <div className="md:col-span-5 flex flex-col gap-6">
              {/* Overall Risk Score Card */}
              <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col gap-4">
                <div>
                  <span className="clinical-label block mb-1">
                    Overall Regimen Risk Score
                  </span>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-[72px] font-normal leading-none font-serif ${getRiskColorClass(report.overall_risk_score)}`}>
                      {report.overall_risk_score.toFixed(0)}
                    </span>
                    <span className="text-xs text-[var(--text-secondary)] font-medium font-mono">/ 100</span>
                  </div>
                </div>

                {/* 6px thin risk bar, rounded ends */}
                <div className="w-full h-1.5 bg-[#E5E7EB] dark:bg-[#2C2C2E] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${report.overall_risk_score}%`,
                      backgroundColor: getRiskColorHex(report.overall_risk_score),
                    }}
                  />
                </div>

                <p className="text-sm text-[var(--text-secondary)] leading-relaxed m-0 font-normal">
                  {report.summary}
                </p>
              </div>

              {/* Risk Contribution Analysis (Shapley chart) */}
              {report.risk_culprit && report.shapley_values && Object.keys(report.shapley_values).length > 0 && (
                <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col gap-4">
                  <div>
                    <h3 className="clinical-label m-0">
                      Risk Contribution
                    </h3>
                    <p className="text-xs text-[var(--text-secondary)] mt-1 m-0 font-normal">
                      Relative interaction risk added by each medication.
                    </p>
                  </div>

                  <div className="flex flex-col gap-3">
                    {Object.entries(report.shapley_values)
                      .sort(([, a], [, b]) => b - a)
                      .map(([name, val]) => {
                        const maxVal = Math.max(...Object.values(report.shapley_values), 1)
                        const pct = Math.max(0, (val / maxVal) * 100)
                        return (
                          <div key={name} className="flex flex-col gap-1">
                            <div className="flex justify-between text-[11px] font-mono">
                              <span className="text-[var(--text-primary)] truncate">{name}</span>
                              <span className="text-[var(--text-secondary)]">{val.toFixed(1)}</span>
                            </div>
                            <div className="w-full h-1.5 bg-[#E5E7EB] dark:bg-[#2C2C2E] rounded-full overflow-hidden">
                              <div
                                className="h-full bg-[var(--text-secondary)] rounded-full"
                                style={{ width: `${pct.toFixed(0)}%` }}
                              />
                            </div>
                          </div>
                        )
                      })}
                  </div>

                  {report.risk_culprit_explanation && (
                    <div className="bg-[var(--bg-app)] rounded p-3 border border-[var(--border-clinical)] mt-1">
                      <p className="text-xs text-[var(--text-primary)] m-0 leading-relaxed font-normal">
                        <span className="font-semibold">Key Culprit: </span>
                        {report.risk_culprit_explanation}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Critical Warfarin Caution Alert */}
              {report.warfarin_warning && (
                <div className="clinical-card border border-[var(--severity-red)] bg-[var(--bg-app)] flex flex-col gap-1.5" id="warfarin-warning">
                  <h4 className="text-xs font-semibold text-[var(--severity-red)] uppercase tracking-wider m-0">
                    Critical Note: Warfarin Regimen Detected
                  </h4>
                  <p className="text-xs text-[var(--text-primary)] leading-relaxed m-0 font-normal">
                    Any modification to your medication list requires immediate pharmacist or physician consultation. Warfarin has over 200 documented drug-drug interactions and is highly sensitive to dosing changes.
                  </p>
                </div>
              )}
            </div>

            {/* Right Column: Interaction checklist & Matrix (col-span-7) */}
            <div className="md:col-span-7 flex flex-col gap-6">
              {/* Interaction Matrix */}
              {report.drug_list.length >= 3 && <InteractionMatrix report={report} />}

              {/* Flagged Interactions List */}
              {report.flagged_interactions.length > 0 && (
                <div className="flex flex-col gap-3">
                  <h2 className="text-sm font-semibold text-[var(--text-primary)] m-0">
                    Flagged Interactions ({report.num_flagged})
                  </h2>
                  <div>
                    {report.flagged_interactions.map(interaction => {
                      const key = `${interaction.drug_a}-${interaction.drug_b}`
                      return (
                        <InteractionCardComponent
                          key={key}
                          interaction={interaction}
                          expanded={expandedCards.has(key)}
                          onToggle={() => toggleCard(key)}
                        />
                      )
                    })}
                  </div>
                </div>
              )}

              {/* No Interactions Safe State */}
              {report.num_flagged === 0 && (
                <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] text-center py-8" id="no-interactions">
                  <h3 className="text-base font-semibold text-[var(--severity-green)] m-0">
                    No Interactions Flagged
                  </h3>
                  <p className="text-xs text-[var(--text-secondary)] max-w-md mx-auto leading-relaxed mt-2 m-0 font-normal">
                    No clinically significant interactions were identified between your {report.drug_list.length} medicines. However, always speak with a medical professional before altering your medication routine.
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Safer Alternatives Recommendation Section (Full-width) */}
          {hasHighRisk && report.risk_culprit && (
            <SaferAlternativesSection
              drugToReplace={report.risk_culprit}
              otherDrugs={otherDrugs}
            />
          )}
        </div>
      )}
    </div>
  )
}
