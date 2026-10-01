import React, { useEffect, useState } from 'react';
import { Check, X, Minus, Loader2, ChevronDown, ExternalLink } from 'lucide-react';
import { durationLabel } from '../lib.js';

function useNow(active) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!active) return undefined;
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

const stageSeconds = (s, now) =>
  s.started_at ? ((s.finished_at ? s.finished_at * 1000 : now) - s.started_at * 1000) / 1000 : null;

function StepIcon({ status }) {
  if (status === 'done') return <span className="step-icon is-done"><Check size={12} strokeWidth={3} /></span>;
  if (status === 'running') return <span className="step-icon is-running"><Loader2 size={13} className="spin" /></span>;
  if (status === 'failed') return <span className="step-icon is-failed"><X size={12} strokeWidth={3} /></span>;
  if (status === 'skipped') return <span className="step-icon is-skipped"><Minus size={12} strokeWidth={3} /></span>;
  return <span className="step-icon is-pending" />;
}

export default function PipelineRunner({ job, defaultOpen = true }) {
  const running = job.status === 'queued' || job.status === 'running';
  const [open, setOpen] = useState(defaultOpen);
  const now = useNow(running);
  const stages = job.stages || [];

  useEffect(() => {
    if (running) setOpen(true);
  }, [running]);

  const doneCount = stages.filter((s) => ['done', 'skipped', 'failed'].includes(s.status)).length;
  const current = stages.find((s) => s.status === 'running');
  const progress = stages.length
    ? Math.round(((doneCount + (current?.progress || 0)) / stages.length) * 100)
    : 0;
  const total = stages.reduce((acc, s) => acc + (stageSeconds(s, now) || 0), 0);

  return (
    <section className="card pipeline">
      <button className="pipeline-head" onClick={() => !running && setOpen((o) => !o)} disabled={running}>
        <div>
          <div className="pipeline-title">
            {running ? 'Analyzing image' : job.status === 'failed' ? 'Analysis failed' : 'Analysis pipeline'}
          </div>
          <div className="pipeline-sub">
            {running
              ? (current ? `Step ${stages.indexOf(current) + 1} of ${stages.length} · ${current.label}` : 'Starting…')
              : `${stages.filter((s) => s.status === 'done').length} steps completed in ${durationLabel(total)}`}
          </div>
        </div>
        <div className="pipeline-head-right">
          <span className="pipeline-pct">{job.status === 'done' ? 100 : progress}%</span>
          {!running && <ChevronDown size={18} className={`chev ${open ? 'is-open' : ''}`} />}
        </div>
      </button>

      <div className="bar"><div className={`bar-fill ${job.status === 'failed' ? 'is-failed' : ''}`} style={{ width: `${job.status === 'done' ? 100 : progress}%` }} /></div>

      {open && (
        <ol className="steps">
          {stages.map((s) => (
            <li key={s.id} className={`step is-${s.status}`}>
              <StepIcon status={s.status} />
              <div className="step-body">
                <div className="step-row">
                  <span className="step-label">{s.label}</span>
                  <span className="step-time">{durationLabel(stageSeconds(s, now))}</span>
                </div>
                <div className="step-detail">{s.detail || s.description}</div>
                {s.status === 'running' && s.progress != null && (
                  <div className="bar bar-thin"><div className="bar-fill" style={{ width: `${Math.round(s.progress * 100)}%` }} /></div>
                )}
                {s.id === 'web_check' && s.status === 'done' && (job.search?.yandex_url || job.search?.search_url) && (
                  <div className="step-links">
                    {job.search.yandex_url && <a href={job.search.yandex_url} target="_blank" rel="noopener noreferrer">Yandex results <ExternalLink size={11} /></a>}
                    {job.search.search_url && <a href={job.search.search_url} target="_blank" rel="noopener noreferrer">Bing results <ExternalLink size={11} /></a>}
                  </div>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}

      {job.status === 'failed' && job.error && <div className="alert alert-danger">{job.error}</div>}
    </section>
  );
}
