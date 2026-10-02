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
      .filter((id) =>
        String(id).toLowerCase().includes(query)
      )
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
      setError("Please select a sample and a drug.");
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
      setError(
        err?.message || "Prediction request failed."
      );
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

      const result = await predictDrugs(
        sampleId,
        selectedDrugIds
      );

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

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">CANXURE</p>

          <h1>Precision Oncology Digital Twin</h1>

          <p className="subtitle">
            Real-data simulation interface powered by ResearchDigitalTwin.
          </p>
        </div>

        <div
          className={`status-pill ${
            health?.status === "ok" ? "ok" : ""
          }`}
        >
          {loading
            ? "Connecting..."
            : health?.status === "ok"
              ? "API Online"
              : "API Offline"}
        </div>
      </header>

      <section className="stats-grid">
        <div className="stat-card">
          <span>Model</span>
          <strong>{health?.model ?? "—"}</strong>
        </div>

        <div className="stat-card">
          <span>Checkpoint</span>
          <strong>{health?.checkpoint ?? "—"}</strong>
        </div>

        <div className="stat-card">
          <span>Real Samples</span>
          <strong>
            {health?.sample_count ?? samples.length}
          </strong>
        </div>

        <div className="stat-card">
          <span>Validated Drugs</span>
          <strong>
            {health?.drug_count ?? drugs.length}
          </strong>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="section-label">
              REAL-DATA PREDICTION
            </p>

            <h2>
              {mode === "single"
                ? "Single Cell-Line / Drug Simulation"
                : "Single Cell-Line / Multi-Drug Simulation"}
            </h2>
          </div>
        </div>

        <div
          style={{
            display: "flex",
            gap: "10px",
            marginBottom: "20px",
            flexWrap: "wrap",
          }}
        >
          <button
            type="button"
            onClick={() => switchMode("single")}
            disabled={loading || predicting}
            style={{
              padding: "10px 16px",
              borderRadius: "10px",
              border: "1px solid rgba(255,255,255,0.15)",
              cursor: "pointer",
              fontWeight: 600,
              opacity: mode === "single" ? 1 : 0.65,
            }}
          >
            Single Drug
          </button>

          <button
            type="button"
            onClick={() => switchMode("multi")}
            disabled={loading || predicting}
            style={{
              padding: "10px 16px",
              borderRadius: "10px",
              border: "1px solid rgba(255,255,255,0.15)",
              cursor: "pointer",
              fontWeight: 600,
              opacity: mode === "multi" ? 1 : 0.65,
            }}
          >
            Multiple Drugs
          </button>
        </div>

        {mode === "single" ? (
          <form
            onSubmit={handlePrediction}
            className="prediction-form"
          >
            <label>
              <span>Cell line</span>

              <select
                value={sampleId}
                onChange={(e) =>
                  setSampleId(e.target.value)
                }
                disabled={
                  loading || !samples.length
                }
              >
                {samples.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Drug</span>

              <select
                value={drugId}
                onChange={(e) =>
                  setDrugId(e.target.value)
                }
                disabled={
                  loading || !drugs.length
                }
              >
                {drugs.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </label>

            <button
              type="submit"
              disabled={predicting || loading}
            >
              {predicting
                ? "Running..."
                : "Run Prediction"}
            </button>
          </form>
        ) : (
          <form
            onSubmit={handleMultiPrediction}
          >
            <label>
              <span>Cell line</span>

              <select
                value={sampleId}
                onChange={(e) =>
                  setSampleId(e.target.value)
                }
                disabled={
                  loading || !samples.length
                }
              >
                {samples.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </label>

            <div style={{ marginTop: "18px" }}>
              <label>
                <span>Search validated drugs</span>

                <input
                  type="text"
                  value={drugSearch}
                  onChange={(e) =>
                    setDrugSearch(e.target.value)
                  }
                  placeholder="Search by drug name..."
                  disabled={
                    loading || !drugs.length
                  }
                  style={{
                    width: "100%",
                    boxSizing: "border-box",
                  }}
                />
              </label>
            </div>

            <div
              style={{
                marginTop: "12px",
                maxHeight: "250px",
                overflowY: "auto",
                border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: "12px",
                padding: "8px",
              }}
            >
              {filteredDrugs.map((drug) => (
                <label
                  key={drug}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    padding: "9px 8px",
                    cursor: "pointer",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={selectedDrugIds.includes(drug)}
                    onChange={() =>
                      toggleDrug(drug)
                    }
                  />

                  <span>{drug}</span>
                </label>
              ))}

              {!filteredDrugs.length && (
                <div style={{ padding: "12px" }}>
                  No matching drugs found.
                </div>
              )}
            </div>

            <div style={{ marginTop: "14px" }}>
              <strong>
                Selected drugs: {selectedDrugIds.length}
              </strong>
            </div>

            {selectedDrugIds.length > 0 && (
              <div
                style={{
                  display: "flex",
                  gap: "8px",
                  flexWrap: "wrap",
                  marginTop: "12px",
                }}
              >
                {selectedDrugIds.map((drug) => (
                  <button
                    key={drug}
                    type="button"
                    onClick={() =>
                      removeSelectedDrug(drug)
                    }
                    disabled={predicting}
                    style={{
                      padding: "7px 10px",
                      borderRadius: "999px",
                      border:
                        "1px solid rgba(255,255,255,0.18)",
                      cursor: "pointer",
                    }}
                  >
                    {drug} ×
                  </button>
                ))}
              </div>
            )}

            <button
              type="submit"
              disabled={
                predicting ||
                loading ||
                !selectedDrugIds.length
              }
              style={{
                marginTop: "18px",
              }}
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
          <div className="error-box">
            {error}
          </div>
        )}

        {prediction && mode === "single" && (
          <div className="result-card">
            <div className="result-header">
              <div>
                <span className="result-label">
                  PREDICTION
                </span>

                <h3>{prediction.drug_id}</h3>

                <p>{prediction.sample_id}</p>
              </div>

              <div className="efficacy-value">
                <span>Predicted log2 AUC</span>

                <strong>
                  {Number(
                    prediction.predicted_log2_auc
                  ).toFixed(4)}
                </strong>
              </div>
            </div>

            <div className="result-grid">
              <div>
                <span>Toxicity</span>
                <strong>
                  {Number(
                    prediction.toxicity
                  ).toFixed(4)}
                </strong>
              </div>

              <div>
                <span>Resistance</span>
                <strong>
                  {Number(
                    prediction.resistance
                  ).toFixed(4)}
                </strong>
              </div>

              <div>
                <span>Clinical logit 0</span>
                <strong>
                  {Number(
                    prediction.clinical_response_logit_0
                  ).toFixed(4)}
                </strong>
              </div>

              <div>
                <span>Clinical logit 1</span>
                <strong>
                  {Number(
                    prediction.clinical_response_logit_1
                  ).toFixed(4)}
                </strong>
              </div>
            </div>

            <p className="model-note">
              Model outputs are shown as returned by the
              Epoch-15 ResearchDigitalTwin. Toxicity,
              resistance, and clinical response logits are
              not presented here as clinically validated
              predictions or treatment recommendations.
            </p>
          </div>
        )}

        {mode === "multi" &&
          multiPredictions.length > 0 && (
            <div style={{ marginTop: "24px" }}>
              <div className="result-card">
                <div className="result-header">
                  <div>
                    <span className="result-label">
                      MULTI-DRUG PREDICTION
                    </span>

                    <h3>{sampleId}</h3>

                    <p>
                      {multiPredictions.length} drugs evaluated
                    </p>
                  </div>
                </div>

                {multiPredictions.map((result) => (
                  <div
                    key={result.drug_id}
                    style={{
                      borderTop:
                        "1px solid rgba(255,255,255,0.10)",
                      paddingTop: "18px",
                      marginTop: "18px",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: "20px",
                        flexWrap: "wrap",
                      }}
                    >
                      <div>
                        <span className="result-label">
                          DRUG
                        </span>

                        <h3>{result.drug_id}</h3>
                      </div>

                      <div className="efficacy-value">
                        <span>
                          Predicted log2 AUC
                        </span>

                        <strong>
                          {Number(
                            result.predicted_log2_auc
                          ).toFixed(4)}
                        </strong>
                      </div>
                    </div>

                    <div className="result-grid">
                      <div>
                        <span>Toxicity</span>

                        <strong>
                          {Number(
                            result.toxicity
                          ).toFixed(4)}
                        </strong>
                      </div>

                      <div>
                        <span>Resistance</span>

                        <strong>
                          {Number(
                            result.resistance
                          ).toFixed(4)}
                        </strong>
                      </div>

                      <div>
                        <span>Clinical logit 0</span>

                        <strong>
                          {Number(
                            result.clinical_response_logit_0
                          ).toFixed(4)}
                        </strong>
                      </div>

                      <div>
                        <span>Clinical logit 1</span>

                        <strong>
                          {Number(
                            result.clinical_response_logit_1
                          ).toFixed(4)}
                        </strong>
                      </div>
                    </div>
                  </div>
                ))}

                <p className="model-note">
                  These values are raw model outputs from
                  the Epoch-15 ResearchDigitalTwin and are
                  not presented as clinically validated
                  treatment recommendations.
                </p>
              </div>
            </div>
          )}
      </section>

      <footer>
        Canxure · Research interface · Epoch-15 ResearchDigitalTwin
      </footer>
    </main>
  );
}
