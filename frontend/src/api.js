// Thin client for the TraceLens investigation API.

export async function startInvestigation(file, { caption = '', forceTrace = false } = {}) {
  const form = new FormData();
  form.append('file', file);
  if (caption) form.append('caption', caption);
  form.append('force_trace', forceTrace ? 'true' : 'false');
  const res = await fetch('/api/investigations', { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Upload failed (${res.status}): ${await res.text()}`);
  return res.json();
}

export async function fetchInvestigation(jobId) {
  const res = await fetch(`/api/investigations/${jobId}`);
  if (!res.ok) throw new Error(`Status check failed (${res.status}): ${await res.text()}`);
  return res.json();
}

// Polls until the job finishes; calls onUpdate with every new snapshot.
// Returns a cancel function.
export function pollInvestigation(jobId, onUpdate, onError, intervalMs = 800) {
  let cancelled = false;
  let failures = 0;
  const tick = async () => {
    if (cancelled) return;
    try {
      const job = await fetchInvestigation(jobId);
      failures = 0;
      if (cancelled) return;
      onUpdate(job);
      if (job.status === 'done' || job.status === 'failed') return;
    } catch (err) {
      failures += 1;
      if (failures >= 5) {
        onError(err);
        return;
      }
    }
    setTimeout(tick, intervalMs);
  };
  tick();
  return () => {
    cancelled = true;
  };
}

export async function fetchHealth() {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error('Backend not reachable');
  return res.json();
}
