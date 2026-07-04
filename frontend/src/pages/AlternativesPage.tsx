import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Loader2, X } from 'lucide-react'
import { getAlternatives } from '../api'
import type { AlternativeRecommendation } from '../api'
import { DrugSearchInput } from '../components/DrugSearchInput'
import { useDrugList } from '../hooks/useDebounce'
import { HelpBulb } from '../components/InfoBulb'

function AlternativeCard({ alt }: { alt: AlternativeRecommendation }) {
  const similarityPct = (alt.similarity_score * 100).toFixed(0)

  return (
    <div
      className="clinical-card border border-[#E2DED8] dark:border-[#2C2C2E] border-l-4 border-l-[var(--severity-green)] p-5 flex flex-col justify-between bg-[var(--bg-card)]"
      id={`alt-${alt.drug_name.replace(/\s+/g, '-').toLowerCase()}`}
    >
      <div>
        <h3 className="text-lg font-medium text-[var(--text-primary)] m-0 font-mono">
          {alt.drug_name}
        </h3>
        <p className="text-xs text-[var(--text-secondary)] mt-2 font-mono">
          Similarity: {similarityPct}%
        </p>
        <p className="text-sm text-[var(--text-primary)] mt-3 leading-relaxed font-normal">
          {alt.mechanism_explanation}
        </p>
      </div>

      <div className="mt-4 pt-3 border-t border-[#E2DED8] dark:border-[#2C2C2E] flex items-center justify-between">
        {alt.risk_reduction_pct > 0 ? (
          <span className="text-xs font-semibold text-[var(--severity-green)]">
            Risk reduced by {Math.abs(alt.risk_reduction_pct).toFixed(0)}%
          </span>
        ) : (
          <span className="text-xs font-semibold text-[var(--severity-green)] font-sans">
            Safe alternative (No interaction)
          </span>
        )}
      </div>
    </div>
  )
}

export function AlternativesPage() {
  const [drugToReplace, setDrugToReplace] = useState('')
  const { drugs: currentDrugs, addDrug, removeDrug } = useDrugList(14)

  const {
    mutate: getAlts,
    data: result,
    isPending,
    error,
  } = useMutation({
    mutationFn: () => getAlternatives(drugToReplace, currentDrugs),
  })

  const canSearch = drugToReplace.trim().length > 1

  return (
    <div className="flex flex-col gap-8 relative">
      <HelpBulb
        purpose="Find molecularly similar drug alternatives to replace a high-risk medication in a patient's regimen while minimizing interaction conflicts."
        inputs="Specify the drug you want to replace in the first search box. Add other current medications you are taking in the second box to compute tailored safety scores. Click Find Alternatives."
        output="Up to 3 recommended alternative drugs shown side-by-side with similarity percentages, exact risk reductions, and clinical explanations of why they are safer."
      />

      {/* Header - DM Serif Display */}
      <div>
        <h1 className="text-3xl font-normal text-[var(--text-primary)] m-0">
          Find a Safer Alternative
        </h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1.5 m-0 font-normal">
          Enter a drug you want to replace and your current medication list to find safer options.
        </p>
      </div>

      {/* Inputs area */}
      <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] flex flex-col gap-4 max-w-[760px] w-full mx-auto">
        {/* Drug to replace */}
        <div className="flex flex-col gap-2">
          <label className="clinical-label">
            Drug to replace
          </label>
          <DrugSearchInput
            id="drug-replace-search"
            onSelect={setDrugToReplace}
            placeholder="Search for the drug you want to replace..."
          />
          {drugToReplace && (
            <div className="prescription-tag flex items-center gap-1.5 text-xs text-[var(--text-primary)] px-2.5 py-1 rounded font-mono w-fit mt-1">
              <span>{drugToReplace}</span>
              <button
                onClick={() => setDrugToReplace('')}
                className="bg-transparent border-0 cursor-pointer flex p-0 text-gray-500 hover:text-red-500 transition-colors"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
        </div>

        {/* Current Regimen */}
        <div className="flex flex-col gap-2">
          <label className="clinical-label">
            Other current medications (improves safety matching)
          </label>
          <DrugSearchInput
            id="current-drug-search"
            onSelect={addDrug}
            placeholder="Add other drugs in regimen..."
          />
          {currentDrugs.length > 0 && (
            <div className="flex flex-wrap gap-2 py-1">
              {currentDrugs.map(d => (
                <div
                  key={d}
                  className="prescription-tag flex items-center gap-1.5 text-xs text-[var(--text-primary)] px-2.5 py-1 rounded font-mono"
                >
                  <span>{d}</span>
                  <button
                    onClick={() => removeDrug(d)}
                    className="bg-transparent border-0 cursor-pointer flex p-0 text-gray-500 hover:text-red-500 transition-colors"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        <button
          id="find-alternatives-btn"
          onClick={() => getAlts()}
          disabled={!canSearch || isPending}
          className="btn-primary w-full py-3 mt-2"
        >
          {isPending ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Finding alternatives...
            </span>
          ) : (
            'Find Alternatives'
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="clinical-card border-red-500 flex flex-col gap-1.5 max-w-[760px] w-full mx-auto">
          <p className="font-semibold text-sm text-[var(--severity-red)] m-0 font-sans">Search Failed</p>
          <p className="text-xs text-[var(--text-secondary)] m-0 font-normal">
            {(error as Error).message || 'API request failed. Verify that the backend server is active.'}
          </p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="flex flex-col gap-6 animate-fade-in" id="alternatives-result">
          {/* Header summary */}
          <div className="border-b border-[#E2DED8] dark:border-[#2C2C2E] pb-4">
            <h2 className="text-lg font-semibold text-[var(--text-primary)] m-0 font-mono">
              Alternatives for {result.drug_to_replace}
            </h2>
            <p className="text-xs text-[var(--text-secondary)] mt-1.5 m-0 font-normal">
              {result.explanation} (Original regimen score: {result.original_risk_score.toFixed(0)})
            </p>
          </div>

          {/* Cards list (3 column grid on desktop, stacked on mobile) */}
          {result.alternatives.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {result.alternatives.map(alt => (
                <AlternativeCard
                  key={alt.drug_name}
                  alt={alt}
                />
              ))}
            </div>
          ) : (
            <div className="clinical-card border-[#E2DED8] dark:border-[#2C2C2E] text-center py-6" id="no-alternatives">
              <p className="text-xs text-[var(--text-secondary)] m-0 font-normal">
                No suitable alternative recommendations found.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Localized Bottom Notice */}
      <div className="p-4 bg-[var(--bg-app)] border border-[#E2DED8] dark:border-[#2C2C2E] rounded" id="alternatives-disclaimer">
        <p className="text-xs text-[var(--text-primary)] leading-relaxed m-0 font-normal">
          <span className="font-semibold">Note: </span>
          Medication substitutions should never be performed without professional medical approval. Suggestions are based on molecular graph similarities and algorithmic safety modeling.
        </p>
      </div>
    </div>
  )
}
