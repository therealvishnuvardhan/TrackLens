import React from 'react';
import { ExternalLink } from 'lucide-react';
import PlatformIcon from './PlatformIcon.jsx';
import { formatDate, isHttp, platformName, pct } from '../lib.js';

function Entry({ node, isTampered, onSelect }) {
  const v = node.version;
  const isUpload = node.kind === 'upload';
  const edited = node.altered_here && node.diff_fraction != null && (!isUpload || isTampered);
  return (
    <li className={`tl-item ${isUpload ? 'is-upload' : ''} ${edited ? 'is-edited' : ''}`}>
      <div className="tl-date">{isUpload ? 'Today' : formatDate(v.timestamp) || '—'}</div>
      <div className="tl-dot" />
      <button className="tl-card" onClick={() => onSelect?.(node)}>
        <div className="tl-thumb">
          {v.image_url && <img src={v.image_url} alt="" loading="lazy" onError={(e) => { e.currentTarget.style.visibility = 'hidden'; }} />}
        </div>
        <div className="tl-body">
          <div className="tl-platform">
            <PlatformIcon url={v.url} isUpload={isUpload} size={14} />
            <span>{isUpload ? 'Your uploaded image' : platformName(v.platform)}</span>
            {node.is_root && <span className="tag tag-origin">Origin</span>}
            {edited && <span className="tag tag-danger">Edited {pct(node.diff_fraction, 1)}</span>}
          </div>
          {!isUpload && v.caption && <div className="tl-caption">{v.caption}</div>}
          {isUpload && <div className="tl-caption">The version you submitted for analysis.</div>}
          {isHttp(v.url) && !isUpload && (
            <a className="tl-link" href={v.url} target="_blank" rel="noopener noreferrer" onClick={(e) => e.stopPropagation()}>
              Open page <ExternalLink size={11} />
            </a>
          )}
        </div>
      </button>
    </li>
  );
}

export default function Timeline({ tree, onSelect }) {
  const isTampered = tree.classification?.verdict === 'TAMPERED';
  const all = Object.values(tree.nodes || {}).filter((n) => n.kind !== 'inferred_original');
  const upload = all.find((n) => n.kind === 'upload');
  const copies = all.filter((n) => n.kind !== 'upload');
  const dated = copies.filter((n) => n.version.timestamp).sort((a, b) => (new Date(a.version.timestamp) - new Date(b.version.timestamp)) || (a.depth - b.depth));
  const undated = copies.filter((n) => !n.version.timestamp);

  if (!copies.length) {
    return <div className="empty">No other copies of this image were found online, so there is no publication timeline.</div>;
  }

  return (
    <div className="timeline">
      <ol className="tl-list">
        {dated.map((n) => <Entry key={n.version.id} node={n} isTampered={isTampered} onSelect={onSelect} />)}
        {upload && <Entry node={upload} isTampered={isTampered} onSelect={onSelect} />}
      </ol>
      {undated.length > 0 && (
        <>
          <div className="tl-section">Publication date not available</div>
          <ol className="tl-list">
            {undated.map((n) => <Entry key={n.version.id} node={n} isTampered={isTampered} onSelect={onSelect} />)}
          </ol>
        </>
      )}
    </div>
  );
}
