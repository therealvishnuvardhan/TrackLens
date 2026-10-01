import React from 'react';
import { pct } from '../lib.js';

function Signal({ s }) {
  const level = s.score >= 0.6 ? 'high' : s.score >= 0.35 ? 'mid' : 'low';
  return (
    <div className="signal">
      <div className="signal-row">
        <span className="signal-name">{s.name}</span>
        <span className={`signal-score is-${level}`}>{pct(s.score)}</span>
      </div>
      <div className="meter"><div className={`meter-fill is-${level}`} style={{ width: pct(s.score) }} /></div>
      <p className="signal-detail">{s.detail}</p>
    </div>
  );
}

export default function ForensicVerdict({ classification, uploadUrl }) {
  if (!classification) return null;
  const images = [
    { url: uploadUrl, label: 'Your image' },
    { url: classification.diff_url, label: 'Changes vs. closest earlier copy', accent: true },
    { url: classification.overlay_url, label: 'Suspicious regions' },
    { url: classification.ela_url, label: 'Error level analysis' },
  ].filter((i) => i.url);
  const steps = classification.manipulation_stages || [];

  return (
    <div className="evidence">
      {images.length > 0 && (
        <div className="evidence-images">
          {images.map((img) => (
            <a key={img.label} className={`evidence-image ${img.accent ? 'is-accent' : ''}`} href={img.url} target="_blank" rel="noopener noreferrer">
              <div className="evidence-image-frame"><img src={img.url} alt={img.label} loading="lazy" /></div>
              <span>{img.label}</span>
            </a>
          ))}
        </div>
      )}

      {steps.length > 0 && (
        <section>
          <h3 className="section-title">Edits detected</h3>
          <ol className="edit-list">
            {steps.map((s, i) => (
              <li key={i}><span className="edit-num">{i + 1}</span><div><strong>{s.label}</strong><p>{s.evidence}</p></div></li>
            ))}
          </ol>
        </section>
      )}

      <section>
        <h3 className="section-title">Signals</h3>
        <div className="signals">
          {(classification.signals || []).map((s) => <Signal key={s.key} s={s} />)}
        </div>
        <p className="footnote">
          {classification.gemini_used ? 'Includes Gemini vision analysis. ' : 'Gemini was unavailable, so this uses local forensics and web comparison only. '}
          Pixel forensics are heuristic; the strongest evidence is a matching copy online that differs from your image.
        </p>
      </section>
    </div>
  );
}
