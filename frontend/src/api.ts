import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000,
})

// ─── Types ────────────────────────────────────────────────────────────────────

export interface InteractionCard {
  drug_a: string
  drug_b: string
  severity: number
  severity_label: 'minor' | 'moderate' | 'major' | 'contraindicated'
  interaction_prob: number
  confidence: number
  mechanism_type: string
  plain_english: string
  clinical_implication: string
  cyp_enzymes: string[]
  is_special_flag: boolean
  support_count: number
  low_data_warning: boolean
  faers_score?: number
  severity_probs?: number[]
}

export interface SpecialFlag {
  flag_type: string
  severity: string
  drugs_involved: string[]
  message: string
  color: string
}

export interface SafetyReport {
  drug_list: string[]
  overall_risk_score: number
  risk_tier: 'safe' | 'review' | 'high' | 'critical'
  risk_tier_label: string
  risk_tier_color: string
  summary: string
  flagged_interactions: InteractionCard[]
  special_flags: SpecialFlag[]
  warfarin_warning: boolean
  risk_culprit: string | null
  risk_culprit_explanation: string
  shapley_values: Record<string, number>
  drug_interaction_counts: Record<string, number>
  num_flagged: number
  num_pairs_checked: number
  interaction_matrix: Record<string, Record<string, number>>
}

export interface PairwiseResponse {
  drug_a: string
  drug_b: string
  interaction_detected: boolean
  severity: number
  severity_label: string
  interaction_prob: number
  confidence: number
  confidence_level: 'high' | 'medium' | 'low'
  mechanism_type: string
  plain_english: string
  clinical_implication: string
  cyp_enzymes: string[]
  is_special_flag: boolean
  support_count: number
  low_data_warning: boolean
  warning_message: string
  gnnexplainer?: Record<string, unknown>
  severity_distribution?: number[]
  severity_distribution_std?: number[]
  drug_a_smiles?: string
  drug_b_smiles?: string
}

export interface AlternativeRecommendation {
  drug_name: string
  drug_id: string
  similarity_score: number
  risk_reduction_pct: number
  total_risk_with_patient: number
  atc_class_match: boolean
  mechanism_explanation: string
  shared_cyp_enzymes: string[]
  confidence: number
}

export interface AlternativeResponse {
  drug_to_replace: string
  current_drugs: string[]
  original_risk_score: number
  alternatives: AlternativeRecommendation[]
  explanation: string
}

export interface DrugSearchResult {
  name: string
  drugbank_id?: string
  atc_class?: string
  categories?: string
  match_score: number
}

// ─── API Functions ────────────────────────────────────────────────────────────

export const analyzePolypharmacy = async (drugs: string[]): Promise<SafetyReport> => {
  const { data } = await api.post<SafetyReport>('/analyze/polypharmacy', { drugs })
  return data
}

export const analyzePairwise = async (drug_a: string, drug_b: string): Promise<PairwiseResponse> => {
  const { data } = await api.post<PairwiseResponse>('/analyze/pairwise', { drug_a, drug_b })
  return data
}

export const getAlternatives = async (
  drug_to_replace: string,
  current_drugs: string[]
): Promise<AlternativeResponse> => {
  const { data } = await api.post<AlternativeResponse>('/recommend/alternative', {
    drug_to_replace,
    current_drugs,
  })
  return data
}

export const searchDrugs = async (q: string, limit = 10): Promise<DrugSearchResult[]> => {
  const { data } = await api.get<{ results: DrugSearchResult[] }>('/drugs/search', {
    params: { q, limit },
  })
  return data.results
}

export const checkHealth = async (): Promise<{ status: string; model_loaded: boolean; drugs_count: number }> => {
  const { data } = await api.get('/') 
  return data
}

export interface DrugNeighbor {
  name: string
  drugbank_id: string
  similarity: number
  smiles?: string
  molecular_weight?: number
  categories?: string
  atc_level1?: string
}

export interface DrugProfile {
  name: string
  drugbank_id?: string
  smiles?: string
  molecular_weight?: number
  description?: string
  mechanism?: string
  atc_level1?: string
  atc_class?: string
  atc_codes?: string
  categories?: string
  groups?: string
}

export const getDrugProfile = async (name: string): Promise<{ drug: DrugProfile; found: boolean }> => {
  const { data } = await api.get<{ drug: DrugProfile; found: boolean }>(`/drugs/${encodeURIComponent(name)}`)
  return data
}

export const getDrugNeighbors = async (name: string): Promise<DrugNeighbor[]> => {
  const { data } = await api.get<{ query: string; neighbors: DrugNeighbor[] }>(
    `/drugs/${encodeURIComponent(name)}/neighbors`
  )
  return data.neighbors
}

