import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Loader2, X } from 'lucide-react'
import { analyzePairwise } from '../api'
import { DrugSearchInput } from '../components/DrugSearchInput'
import { HelpBulb } from '../components/InfoBulb'

const SEV_STYLE = {
  minor:           { border: 'var(--text-secondary)', text: 'var(--text-secondary)' },
  moderate:        { border: 'var(--severity-yellow)', text: 'var(--severity-yellow)' },
  major:           { border: 'var(--severity-amber)',  text: 'var(--severity-amber)' },
  contraindicated: { border: 'var(--severity-red)',    text: 'var(--severity-red)' },
}

// Helper to strip low-data warnings from clinical implications
const cleanClinicalImplication = (text: string) => {
  if (!text) return ''
  return text.replace(/⚠️\s*Limited data \(\d+ source\(s\)\)\. Interaction status uncertain\. Consult a pharmacist for personalized advice\.\s*/gi, '')
}

function StructureRender({ smiles, name }: { smiles?: string; name: string }) {
  const [imgError, setImgError] = useState(false)

  if (!smiles || imgError) {
    return (
      <div className="w-40 h-40 flex items-center justify-center border border-[#E2DED8] dark:border-[#2C2C2E] rounded text-[11px] text-[var(--text-secondary)] text-center p-2">
        Structure not available for {name}
      </div>
    )
  }

  const imgUrl = `https://cactus.nci.nih.gov/chemical/structure/${encodeURIComponent(smiles)}/image?format=gif&width=200&height=200`

  return (
    <div className="w-40 h-40 flex flex-col items-center justify-center bg-white p-2 rounded border border-[#E2DED8] dark:border-[#2C2C2E]">
      <img
        src={imgUrl}
        alt={`Chemical structure of ${name}`}
        className="w-full h-full object-contain mix-blend-multiply dark:invert dark:contrast-[1.2] transition-all"
        onError={() => setImgError(true)}
      />
    </div>
  )
}

export function PairwiseLookupPage() {
  const [drugA, setDrugA] = useState('')
  const [drugB, setDrugB] = useState('')

  const { mutate: analyze, data: result, isPending, error } = useMutation({
    mutationFn: () => analyzePairwise(drugA, drugB),
  })

  const canAnalyze = drugA.trim().length > 1 && drugB.trim().length > 1 && drugA.toLowerCase() !== drugB.toLowerCase()
  const sev = result ? (SEV_STYLE[result.severity_label as keyof typeof SEV_STYLE] || SEV_STYLE.minor) : null

  return (
    <div className="flex flex-col gap-8 relative">
      <HelpBulb
        purpose="Compare two specific drugs to check if they have a clinical interaction, explain the physiological mechanism, and view GNN Explainer attributions."
        inputs="Select the first drug in the Drug A field, and the second drug in the Drug B field using the autocomplete inputs. Click the Check Interaction button."
        output="A text severity badge, a detailed clinical mechanism explanation card, side-by-side molecular structure diagrams, confidence score percentage, and GNN model explanation footnotes."
      />

      {/* Header - DM Serif Display */}
      <div>
        <h1 className="text-3xl font-normal text-[var(--text-primary)] m-0">
          Check Two Drugs
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1.5 m-0 font-normal">
          Look up the interaction between any two specific drugs.
        </p>
      </div>

      {/* Inputs panel - capped width */}
      <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col gap-4 max-w-[760px] w-full mx-auto">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="flex flex-col gap-2">
            <label className="clinical-label">
              Drug A
            </label>
            <DrugSearchInput
              id="drug-a-search"
              onSelect={setDrugA}
              placeholder="Type drug A name..."
            />
            {drugA && (
              <div className="prescription-tag flex items-center gap-1.5 text-xs text-[var(--text-primary)] px-2.5 py-1 rounded font-mono w-fit mt-1">
                <span>{drugA}</span>
                <button
                  onClick={() => setDrugA('')}
                  className="bg-transparent border-0 cursor-pointer flex p-0 text-gray-500 hover:text-red-500 transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>

          <div className="flex flex-col gap-2">
            <label className="clinical-label">
              Drug B
            </label>
            <DrugSearchInput
              id="drug-b-search"
              onSelect={setDrugB}
              placeholder="Type drug B name..."
            />
            {drugB && (
              <div className="prescription-tag flex items-center gap-1.5 text-xs text-[var(--text-primary)] px-2.5 py-1 rounded font-mono w-fit mt-1">
                <span>{drugB}</span>
                <button
                  onClick={() => setDrugB('')}
                  className="bg-transparent border-0 cursor-pointer flex p-0 text-gray-500 hover:text-red-500 transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        </div>

        <button
          id="pairwise-analyze-btn"
          onClick={() => analyze()}
          disabled={!canAnalyze || isPending}
          className="btn-primary w-full py-3 mt-2"
        >
          {isPending ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Checking interaction...
            </span>
          ) : (
            'Check Interaction'
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="clinical-card border-red-500 flex flex-col gap-1.5 max-w-[760px] w-full mx-auto">
          <p className="font-semibold text-sm text-[var(--severity-red)] m-0 font-sans">Failed to Check</p>
          <p className="text-xs text-[var(--text-secondary)] m-0 font-normal">
            {(error as Error).message || 'API request failed. Verify that the backend server is active.'}
          </p>
        </div>
      )}

      {/* Results details */}
      {result && sev && (
        <div className="flex flex-col gap-6 animate-fade-in" id="pairwise-result">
          {/* Header Verdict Badge & Drug Pairs */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#E2DED8] dark:border-[#2C2C2E] pb-4">
            <div>
              <h2 className="text-lg font-semibold text-[var(--text-primary)] m-0 font-mono">
                {result.drug_a} + {result.drug_b}
              </h2>
              <span className="text-xs text-[var(--text-secondary)] mt-0.5 block font-mono">
                Mechanism: {result.mechanism_type.replace(/_/g, ' ')}
              </span>
            </div>

            {/* Severity text badge (White / Card background with colored border) */}
            <div className="flex items-center gap-3 shrink-0">
              {result.interaction_prob !== undefined && (
                <span className="text-xs text-[var(--text-secondary)] font-mono">
                  Probability: {(result.interaction_prob * 100).toFixed(0)}%
                </span>
              )}
              <span
                className="sev-badge"
                style={{
                  borderColor: sev.border,
                  color: sev.text,
                }}
              >
                {result.severity_label}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-start">
            {/* Left Column (col-span-7) */}
            <div className="md:col-span-7 flex flex-col gap-6">
              {result.warning_message && !result.warning_message.toLowerCase().includes('limited data') && (
                <div className="p-3 bg-[var(--bg-app)] border border-[#E2DED8] dark:border-[#2C2C2E] text-xs text-[var(--text-primary)] leading-normal font-normal">
                  Warning: {result.warning_message}
                </div>
              )}

              {/* Mechanism Explanation Card */}
              <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] p-6 flex flex-col gap-3">
                <h3 className="clinical-label m-0">
                  Mechanism Explanation
                </h3>
                <p className="text-[15px] text-[var(--text-primary)] leading-relaxed m-0 font-normal">
                  {result.plain_english}
                </p>
                {result.clinical_implication && (
                  <div className="mt-2 pt-3 border-t border-[#E2DED8] dark:border-[#2C2C2E]">
                    <p className="text-xs text-[var(--text-secondary)] m-0 leading-normal">
                      <span className="font-semibold text-[var(--text-primary)]">Clinical Implication: </span>
                      {cleanClinicalImplication(result.clinical_implication)}
                    </p>
                  </div>
                )}
              </div>

              {/* Confidence Score & GNN Explainer Footnote */}
              <div className="flex flex-col gap-3 pt-2">
                <span className="text-xs text-[var(--text-secondary)] font-mono">
                  Model confidence: {(result.confidence * 100).toFixed(0)}%
                </span>

                {result.gnnexplainer && (
                  <aside className="p-4 bg-[var(--bg-app)] border border-[#E2DED8] dark:border-[#2C2C2E] text-xs text-[var(--text-primary)] rounded" id="gnnexplainer-result">
                    <p className="clinical-label m-0 mb-1.5">
                      GNNExplainer Footnote
                    </p>
                    <p className="m-0 leading-relaxed font-normal">
                      {(result.gnnexplainer.explanation_text as string) || 'GNNExplainer analysis complete.'}
                    </p>
                  </aside>
                )}
              </div>
            </div>

            {/* Right Column (col-span-5) */}
            <div className="md:col-span-5 flex flex-col gap-6">
              {/* Molecular Structure Comparison Card */}
              <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] p-6 flex flex-col gap-4">
                <h3 className="clinical-label m-0">
                  Molecular Structure Comparison
                </h3>
                <div className="flex flex-row items-center justify-around gap-6 flex-wrap">
                  <div className="flex flex-col gap-2 items-center">
                    <StructureRender smiles={result.drug_a_smiles} name={result.drug_a} />
                    <span className="text-xs text-[var(--text-secondary)] font-mono font-semibold">{result.drug_a}</span>
                  </div>
                  <div className="flex flex-col gap-2 items-center">
                    <StructureRender smiles={result.drug_b_smiles} name={result.drug_b} />
                    <span className="text-xs text-[var(--text-secondary)] font-mono font-semibold">{result.drug_b}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
