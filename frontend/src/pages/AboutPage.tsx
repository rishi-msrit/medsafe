import { Mail } from 'lucide-react'

const DATASETS = [
  {
    name: 'DrugBank',
    desc: 'Curated dataset of FDA-approved drugs, molecular structures, and established drug-drug interaction mechanism logs.',
    url: 'https://go.drugbank.com'
  },
  {
    name: 'TWOSIDES',
    desc: 'Database of multi-drug side effects and adverse interaction reactions mined from medical health records.',
    url: 'https://tatonettilab.org/resources/twosides/'
  },
  {
    name: 'OGBL-DDI',
    desc: 'Standardized drug-drug interaction benchmark dataset curated by the Open Graph Benchmark for graph neural network testing.',
    url: 'https://ogb.stanford.edu/docs/linkprop/#ogbl-ddi'
  }
]

export function AboutPage() {
  return (
    <div className="flex flex-col gap-8 relative">

      {/* Header - DM Serif Display */}
      <div>
        <h1 className="text-3xl font-normal text-[var(--text-primary)] m-0">
          About MedSafe
        </h1>
      </div>

      {/* About Me Section */}
      <section className="flex flex-col gap-4 border-b border-[#E2DED8] dark:border-[#2C2C2E] pb-6">
        <h2 className="clinical-label m-0">
          About Me
        </h2>
        <div className="flex flex-col gap-3">
          <p className="text-sm text-[var(--text-primary)] leading-relaxed m-0 font-normal">
            Hi! I am Rishi, a third-year Electronics student at RIT, Bangalore. I developed MedSafe as an independent ML research project to explore the power of graph neural networks applied to clinical pharmacology and drug safety modeling.
          </p>
          <div className="flex flex-col gap-2 bg-[var(--bg-card)] border border-[#E2DED8] dark:border-[#2C2C2E] p-4 rounded mt-1">
            <span className="clinical-label block mb-1">Technical Implementation Details</span>
            <ul className="text-xs text-[var(--text-primary)] leading-relaxed pl-4 m-0 space-y-1.5 font-normal list-disc">
              <li>
                <strong className="font-semibold">Relational GNN Modeling:</strong> Utilized a multi-layer Relational Graph Convolutional Network (R-GCN) framework to learn drug representations and classify pairwise interactions directly from molecular graphs.
              </li>
              <li>
                <strong className="font-semibold">Explainability Attribution:</strong> Calculated Shapley values to identify specific culprit drugs contributing to overall polypharmacy risk.
              </li>
              <li>
                <strong className="font-semibold">Tanimoto Similarity Matching:</strong> Queried structurally-similar alternative drugs via fingerprinted Tanimoto coefficient metrics.
              </li>
              <li>
                <strong className="font-semibold">Explainable Predictions:</strong> Generated GNNExplainer features mapping molecular substructure subgraphs to predicted interactions.
              </li>
            </ul>
          </div>
          <div className="flex flex-col items-center gap-3 mt-6 border-t border-[#E2DED8] dark:border-[#2C2C2E] pt-6">
            <p className="text-sm text-[var(--text-secondary)] m-0 font-medium text-center">
              Wanna connect? Got any suggestions? Found a bug?
            </p>
            <div className="flex items-center justify-center gap-4 flex-wrap">
              <a
                href="https://github.com/rishi-msrit"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded bg-[var(--bg-tag)] hover:bg-[var(--border-clinical)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all border border-[var(--border-input)] cursor-pointer flex items-center justify-center"
                title="GitHub"
              >
                <img
                  src="/github_forlight.png"
                  alt="GitHub"
                  className="w-5 h-5 object-contain"
                />
              </a>
              <a
                href="https://www.linkedin.com/in/rishi-msrit/"
                target="_blank"
                rel="noopener noreferrer"
                className="p-2 rounded bg-[var(--bg-tag)] hover:bg-[var(--border-clinical)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all border border-[var(--border-input)] cursor-pointer flex items-center justify-center"
                title="LinkedIn"
              >
                <img
                  src="/linkedin_forlight.png"
                  alt="LinkedIn"
                  className="w-5 h-5 object-contain"
                />
              </a>
              <a
                href="mailto:rishi.msrit@gmail.com"
                className="p-2 rounded bg-[var(--bg-tag)] hover:bg-[var(--border-clinical)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-all border border-[var(--border-input)] cursor-pointer flex items-center justify-center"
                title="Email"
              >
                <Mail className="w-4.5 h-4.5" />
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Project Section */}
      <section className="flex flex-col gap-4">
        <h2 className="clinical-label m-0">
          Project Overview
        </h2>
        <p className="text-sm text-[var(--text-primary)] leading-relaxed m-0 font-normal">
          MedSafe is a clinical assistant prototype designed to help pharmacists and patients analyze safety risks in polypharmacy regimens (taking five or more medicines concurrently). In complex drug regimens, traditional search lookup databases often miss multi-drug interactions or structural interactions that haven't been cataloged in static charts. This tool helps identify conflicts by analyzing the molecular structure of drugs to predict side effects.
        </p>
        <p className="text-sm text-[var(--text-primary)] leading-relaxed m-0 font-normal">
          By representing each medication as a molecular graph and utilizing relational graph models, MedSafe predicts potential interaction severity levels. It acts as an early warning system to help users recognize dangerous combinations, verify alternative options, and consult medical practitioners with clear, structure-backed clinical questions.
        </p>
      </section>

      {/* How It Works Section */}
      <section className="flex flex-col gap-4">
        <h2 className="clinical-label m-0">
          How It Works
        </h2>
        <div className="flex flex-col gap-4 pl-1">
          {[
            'You enter your medications',
            'The system builds a molecular graph for each drug',
            'A Graph Neural Network analyzes chemical structure and known interactions',
            'The model predicts interaction severity and explains which molecular features caused it',
            'A risk score is computed for your full medication list',
            'Safer alternatives are suggested based on molecular similarity'
          ].map((step, idx) => (
            <div key={idx} className="flex gap-4 items-start text-sm">
              <span className="font-mono font-semibold text-[var(--text-secondary)] w-6 text-right shrink-0">
                {idx + 1}.
              </span>
              <p className="m-0 text-[var(--text-primary)] font-normal leading-normal">
                {step}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Disclaimer Section - Bordered warm card style */}
      <section className="p-5 bg-[var(--bg-card)] border border-[#E2DED8] dark:border-[#2C2C2E] rounded">
        <p className="text-xs text-[var(--text-primary)] leading-relaxed m-0 font-normal">
          MedSafe is an educational resource and not a medical device, is not validated for clinical use, and should not be used to make any medical decisions. Always consult a licensed pharmacist or physician before making changes to your medications.
        </p>
      </section>

      {/* Dataset Sources Section */}
      <section className="flex flex-col gap-4">
        <h2 className="clinical-label m-0">
          Dataset Sources
        </h2>
        <div className="overflow-x-auto border border-[#E2DED8] dark:border-[#2C2C2E] rounded">
          <table className="w-full text-xs font-medium border-collapse text-left bg-[var(--bg-card)]">
            <thead>
              <tr className="border-b border-[#E2DED8] dark:border-[#2C2C2E] bg-[var(--bg-app)]">
                <th className="p-3 font-semibold text-[var(--text-primary)]">Dataset</th>
                <th className="p-3 font-semibold text-[var(--text-primary)]">Description</th>
                <th className="p-3 font-semibold text-[var(--text-primary)]">Source URL</th>
              </tr>
            </thead>
            <tbody>
              {DATASETS.map((d, idx) => (
                <tr key={idx} className="border-b border-[#E2DED8] dark:border-[#2C2C2E] last:border-0 hover:bg-[var(--bg-app)] transition-colors">
                  <td className="p-3 font-mono font-semibold text-[var(--text-primary)]">{d.name}</td>
                  <td className="p-3 text-[var(--text-secondary)] font-normal leading-relaxed">{d.desc}</td>
                  <td className="p-3 font-mono">
                    <a
                      href={d.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[var(--accent)] hover:underline no-underline"
                    >
                      Source Link
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
