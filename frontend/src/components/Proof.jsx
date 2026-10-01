import React from 'react';
import { ExternalLink } from 'lucide-react';

const STRENGTH = {
  decisive: { label: 'Direct proof', note: 'A reference shows different content' },
  strong: { label: 'Strong evidence', note: 'A trained detector localises an edit' },
  supporting: { label: 'Supporting signal', note: 'Heuristic; not proof on its own' },
};

export default function Proof({ proof, verdict }) {
  if (!proof?.length) {
    return (
      <div className="empty">
        <p className="empty-title">No evidence of editing was found.</p>
        <p>
          {verdict === 'TAMPERED'
            ? 'The verdict rests on combined weak signals only — treat it with caution.'
            : 'The image matches its copies online, its embedded data is consistent, and no detector localised an edit. This does not prove it is untouched — only that nothing was found.'}
        </p>
      </div>
    );
  }

  return (
    <div className="proof">
      {proof.map((p, i) => (
        <article key={`${p.key}-${i}`} style={{ animationDelay: `${i * 70}ms` }} className={`proof-item is-${p.strength}`}>
          <header className="proof-head">
            <span className={`pill pill-${p.strength}`}>{STRENGTH[p.strength]?.label || p.strength}</span>
            <span className="proof-note">{STRENGTH[p.strength]?.note}</span>
          </header>
          <h3 className="proof-title">{p.title}</h3>
          <p className="proof-detail">{p.detail}</p>
          {p.images?.length > 0 && (
            <div className={`proof-images n-${Math.min(p.images.length, 3)}`}>
              {p.images.map((img) => (
                <a key={img.url} className="proof-image" href={img.url} target="_blank" rel="noopener noreferrer">
                  <div className="proof-image-frame"><img src={img.url} alt={img.label} loading="lazy" /></div>
                  <span>{img.label}</span>
                </a>
              ))}
            </div>
          )}
          {p.links?.length > 0 && (
            <div className="proof-links">
              {p.links.map((l) => (
                <a key={l.url} href={l.url} target="_blank" rel="noopener noreferrer">
                  {l.label} <ExternalLink size={12} />
                </a>
              ))}
            </div>
          )}
        </article>
      ))}
    </div>
  );
}
