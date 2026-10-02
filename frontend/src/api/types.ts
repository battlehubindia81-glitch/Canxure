export interface HealthResponse {
  service: string;
  status: string;
  model: string;
  checkpoint: string;
  sample_count: number;
  drug_count: number;
}

export interface PredictionResult {
  sample_id: string;
  drug_id: string;
  predicted_log2_auc: number;
  toxicity: number;
  resistance: number;
  clinical_response_logit_0: number;
  clinical_response_logit_1: number;
  checkpoint_epoch: number;
}

export interface MultiPredictionResponse {
  results: PredictionResult[];
}

export interface SampleResponse {
  sample_ids: string[];
}

export interface DrugResponse {
  drug_ids: string[];
}
