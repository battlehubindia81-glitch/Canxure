import React, { useEffect, useMemo, useState } from "react";
import {
  getHealth,
  getSamples,
  getDrugs,
  predictPair,
  predictDrugs,
} from "./api/client";

export default function App() {
  const [health, setHealth] = useState(null);
  const [samples, setSamples] = useState([]);
  const [drugs, setDrugs] = useState([]);

  const [sampleId, setSampleId] = useState("");
  const [drugId, setDrugId] = useState("");

  const [prediction, setPrediction] = useState(null);
  const [multiPredictions, setMultiPredictions] = useState([]);

  const [selectedDrugIds, setSelectedDrugIds] = useState([]);
  const [drugSearch, setDrugSearch] = useState("");

  const [mode, setMode] = useState("single");

  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        setLoading(true);
        setError("");

        const [healthData, sampleData, drugData] = await Promise.all([
          getHealth(),
          getSamples(),
          getDrugs(),
        ]);

        setHealth(healthData);

        const sampleIds = sampleData?.sample_ids ?? [];
        const drugIds = drugData?.drug_ids ?? [];

        setSamples(sampleIds);
        setDrugs(drugIds);

        if (sampleIds.length) {
          setSampleId(sampleIds[0]);
        }

        if (drugIds.length) {
          setDrugId(drugIds[0]);
        }
      } catch (err) {
        setError(
          err?.message || "Failed to connect to Canxure API."
        );
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);

  const filteredDrugs = useMemo(() => {
    const query = drugSearch.trim().toLowerCase();

    if (!query) {
      return drugs.slice(0, 40);
    }

    return drugs
      .filter((id) => String(id).toLowerCase().includes(query))
      .slice(0, 40);
  }, [drugs, drugSearch]);

  function toggleDrug(drug) {
    setSelectedDrugIds((current) => {
      if (current.includes(drug)) {
        return current.filter((id) => id !== drug);
      }

      return [...current, drug];
    });
  }

  function removeSelectedDrug(drug) {
    setSelectedDrugIds((current) =>
      current.filter((id) => id !== drug)
    );
  }

  function switchMode(nextMode) {
    setMode(nextMode);
    setError("");

    if (nextMode === "single") {
      setMultiPredictions([]);
    } else {
      setPrediction(null);
    }
  }

  async function handlePrediction(event) {
    event.preventDefault();

    if (!sampleId || !drugId) {
      setError("Please select a cell line and a drug.");
      return;
    }

    try {
      setPredicting(true);
      setError("");
      setPrediction(null);

      const result = await predictPair(sampleId, drugId);
      setPrediction(result);
    } catch (err) {
      setPrediction(null);
      setError(err?.message || "Prediction request failed.");
    } finally {
      setPredicting(false);
    }
  }

  async function handleMultiPrediction(event) {
    event.preventDefault();

    if (!sampleId) {
      setError("Please select a cell line.");
      return;
    }

    if (!selectedDrugIds.length) {
      setError("Please select at least one drug.");
      return;
    }

    try {
      setPredicting(true);
      setError("");
      setPrediction(null);
      setMultiPredictions([]);

      const result = await predictDrugs(sampleId, selectedDrugIds);
      setMultiPredictions(result?.results ?? []);
    } catch (err) {
      setMultiPredictions([]);
      setError(
        err?.message || "Multi-drug prediction request failed."
      );
    } finally {
      setPredicting(false);
    }
  }

  function renderMetrics(result) {
    return (
      <div className="result-grid">
        <div className="metric-card">
          <span>Toxicity</span>
          <strong>{Number(result.toxicity).toFixed(4)}</strong>
        </div>

        <div className="metric-card">
          <span>Resistance</span>
          <strong>{Number(result.resistance).toFixed(4)}</strong>
        </div>

        <div className="metric-card">
          <span>Clinical logit 0</span>
          <strong>
            {Number(result.clinical_response_logit_0).toFixed(4)}
          </strong>
        </div>

        <div className="metric-card">
          <span>Clinical logit 1</span>
          <strong>
            {Number(result.clinical_response_logit_1).toFixed(4)}
          </strong>
        </div>
      </div>
    );
  }

  function renderPrediction(result) {
    return (
      <>
        <div className="result-header">
          <div className="result-identity">
            <span className="result-label">PREDICTION</span>
            <h3>{result.drug_id}</h3>
            <p>{result.sample_id}</p>
          </div>

          <div className="efficacy-value">
            <span>Predicted log2 AUC</span>
            <strong>
              {Number(result.predicted_log2_auc).toFixed(4)}
            </strong>
          </div>
        </div>

        {renderMetrics(result)}
      </>
    );
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="brand-lockup">
          <h1 className="brand-title">CANXURE</h1>
          <p className="subtitle">Precision Oncology Digital Twin</p>
          <p className="hero-description">
            Real-data research interface powered by ResearchDigitalTwin.
          </p>
        </div>
      </header>

      <div className="mode-switch" role="group" aria-label="Prediction mode">
        <button
          type="button"
          className={`mode-button ${mode === "single" ? "active" : ""}`}
          onClick={() => switchMode("single")}
          disabled={loading || predicting}
          aria-pressed={mode === "single"}
        >
          Single Drug
        </button>

        <button
          type="button"
          className={`mode-button ${mode === "multi" ? "active" : ""}`}
          onClick={() => switchMode("multi")}
          disabled={loading || predicting}
          aria-pressed={mode === "multi"}
        >
          Multiple Drugs
        </button>
      </div>

      <section className="panel" aria-labelledby="prediction-heading">
        <div className="panel-heading">
          <p className="section-label">REAL-DATA PREDICTION</p>
          <h2 id="prediction-heading">
            {mode === "single"
              ? "Single Cell-Line / Drug Simulation"
              : "Single Cell-Line / Multi-Drug Simulation"}
          </h2>
        </div>

        {mode === "single" ? (
          <form onSubmit={handlePrediction} className="prediction-form">
            <label className="form-field">
              <span>Select Cell Line</span>
              <select
                value={sampleId}
                onChange={(event) => setSampleId(event.target.value)}
                disabled={loading || !samples.length}
              >
                {samples.map((id) => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
            </label>

            <label className="form-field">
              <span>Select Drug</span>
              <select
                value={drugId}
                onChange={(event) => setDrugId(event.target.value)}
                disabled={loading || !drugs.length}
              >
                {drugs.map((id) => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
            </label>

            <button
              type="submit"
              className="primary-button"
              disabled={predicting || loading || !samples.length || !drugs.length}
            >
              {predicting ? "Running..." : "Run Prediction"}
            </button>
          </form>
        ) : (
          <form onSubmit={handleMultiPrediction} className="multi-form">
            <label className="form-field">
              <span>Select Cell Line</span>
              <select
                value={sampleId}
                onChange={(event) => setSampleId(event.target.value)}
                disabled={loading || !samples.length}
              >
                {samples.map((id) => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
            </label>

            <label className="form-field search-field">
              <span>Search Validated Drugs</span>
              <input
                type="text"
                value={drugSearch}
                onChange={(event) => setDrugSearch(event.target.value)}
                placeholder="Search by drug name..."
                disabled={loading || !drugs.length}
              />
            </label>

            <div className="drug-picker" aria-label="Available drugs">
              {filteredDrugs.map((drug) => (
                <label key={drug} className="drug-option">
                  <input
                    type="checkbox"
                    checked={selectedDrugIds.includes(drug)}
                    onChange={() => toggleDrug(drug)}
                    disabled={loading || predicting}
                  />
                  <span>{drug}</span>
                </label>
              ))}

              {!filteredDrugs.length && (
                <p className="empty-search">No matching drugs found.</p>
              )}
            </div>

            <div className="selected-summary">
              <strong>Selected drugs: {selectedDrugIds.length}</strong>
            </div>

            {selectedDrugIds.length > 0 && (
              <div className="selected-drugs">
                {selectedDrugIds.map((drug) => (
                  <button
                    key={drug}
                    type="button"
                    className="selected-drug-chip"
                    onClick={() => removeSelectedDrug(drug)}
                    disabled={predicting}
                    aria-label={`Remove ${drug}`}
                  >
                    {drug} <span aria-hidden="true">×</span>
                  </button>
                ))}
              </div>
            )}

            <button
              type="submit"
              className="primary-button multi-submit"
              disabled={predicting || loading || !selectedDrugIds.length}
            >
              {predicting
                ? "Running..."
                : `Run Multi-Drug Prediction${
                    selectedDrugIds.length
                      ? ` (${selectedDrugIds.length})`
                      : ""
                  }`}
            </button>
          </form>
        )}

        {error && (
          <div className="error-box" role="alert">
            {error}
          </div>
        )}

        <aside className="research-disclaimer">
          <h3>Please Read Carefully</h3>
          <p>
            This model is intended strictly for research purposes and was
            trained using data from the Genomics of Drug Sensitivity in
            Cancer (GDSC) database. It has not undergone clinical validation
            and is not approved for clinical use or treatment decisions.
          </p>
        </aside>

        {prediction && mode === "single" && (
          <div className="result-card">
            {renderPrediction(prediction)}
            <p className="model-note">
              Outputs are shown as returned by the Epoch-15 ResearchDigitalTwin.
              Toxicity, resistance, and clinical-response logits are not
              clinically validated predictions or treatment recommendations.
            </p>
          </div>
        )}

        {mode === "multi" && multiPredictions.length > 0 && (
          <div className="result-card multi-results">
            <div className="multi-result-heading">
              <span className="result-label">MULTI-DRUG PREDICTION</span>
              <h3>{sampleId}</h3>
              <p>{multiPredictions.length} drugs evaluated</p>
            </div>

            {multiPredictions.map((result) => (
              <article key={result.drug_id} className="multi-result-item">
                {renderPrediction(result)}
              </article>
            ))}

            <p className="model-note">
              These outputs are generated by the Epoch-15 ResearchDigitalTwin.
              They are not clinically validated treatment recommendations.
            </p>
          </div>
        )}
      </section>

      <div className="api-status-area">
        <div
          className={`status-pill ${
            health?.status === "ok" ? "ok" : "offline"
          }`}
          aria-live="polite"
        >
          <span className="status-dot" aria-hidden="true" />
          {loading
            ? "Connecting to API..."
            : health?.status === "ok"
              ? "API Online"
              : "API Offline"}
        </div>
      </div>

      <footer className="site-footer">
        <div className="footer-brand">
          <strong>CANXURE</strong>
          <span>Control developed</span>

          <div className="footer-socials" aria-label="Social media">
            <span className="social-mark instagram-mark" role="img" aria-label="Instagram">
              ◎
            </span>
            <span className="social-mark x-mark" role="img" aria-label="X">
              𝕏
            </span>
            <span className="social-mark linkedin-mark" role="img" aria-label="LinkedIn">
              in
            </span>
          </div>
        </div>

        <div className="footer-data">
          <span className="footer-data-label">Data Used:</span>
          <a
            href="https://depmap.org/portal/data_page/?tab=allData"
            target="_blank"
            rel="noopener noreferrer"
          >
            deepmap.org
          </a>
        </div>
      </footer>

      <p className="page-note">
        CANXURE · Research interface · Not validated for clinical use
      </p>
    </main>
  );
}
