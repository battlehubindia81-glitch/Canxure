const DEFAULT_BASE_URL = "/api";

function getBaseUrl() {
  return (
    import.meta.env.VITE_CANXURE_API_URL ||
    DEFAULT_BASE_URL
  ).replace(/\/$/, "");
}

async function parseResponse(response) {
  const text = await response.text();

  let data = null;

  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new Error(
      `Canxure API returned non-JSON data (HTTP ${response.status})`
    );
  }

  if (!response.ok) {
    const message =
      data?.detail ||
      data?.message ||
      `Canxure API request failed (HTTP ${response.status})`;

    throw new Error(message);
  }

  return data;
}

export async function getHealth() {
  const response = await fetch(`${getBaseUrl()}/health`);
  return parseResponse(response);
}

export async function getSamples() {
  const response = await fetch(`${getBaseUrl()}/samples`);
  return parseResponse(response);
}

export async function getDrugs() {
  const response = await fetch(`${getBaseUrl()}/drugs`);
  return parseResponse(response);
}

export async function predictPair(sampleId, drugId) {
  const params = new URLSearchParams({
    sample_id: sampleId,
    drug_id: drugId,
  });

  const response = await fetch(
    `${getBaseUrl()}/predict?${params.toString()}`
  );

  return parseResponse(response);
}

export async function predictDrugs(sampleId, drugIds) {
  const response = await fetch(`${getBaseUrl()}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      sample_id: sampleId,
      drug_ids: drugIds,
    }),
  });

  return parseResponse(response);
}
