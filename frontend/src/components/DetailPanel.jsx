import React, { useEffect } from 'react';
import { X, ExternalLink } from 'lucide-react';
import PlatformIcon from './PlatformIcon.jsx';
import { formatDate, isHttp, platformName, pct, CATEGORY_LABEL } from '../lib.js';

function Row({ label, children }) {
  if (children == null || children === false || children === '') return null;
  return (
    <div className="kv">
      <span className="kv-k">{label}</span>
      <span className="kv-v">{children}</span>
    </div>
  );
}

function NodeDetail({ node, tree }) {
  const v = node.version;
  const isUpload = node.kind === 'upload';
  const parent = node.parent_id ? tree.nodes[node.parent_id] : null;
  const fa = isUpload ? node.forensic_analysis : null;
  return (
    <>
      {v.image_url && <div className="panel-img"><img src={v.image_url} alt="" /></div>}
      <div className="panel-title">
        <PlatformIcon url={v.url} isUpload={isUpload} size={18} />
        <h2>{isUpload ? 'Your image' : node.kind === 'inferred_original' ? 'Original (not found online)' : platformName(v.platform)}</h2>
      </div>
      {v.caption && !isUpload && <p className="panel-caption">{v.caption}</p>}
      <div className="kv-list">
        <Row label="Type">{CATEGORY_LABEL[v.category] || null}</Row>
        <Row label="Published">{isUpload ? null : formatDate(v.timestamp) || 'Not available'}</Row>
        <Row label="Resolution">{v.dimensions ? `${v.dimensions[0]} × ${v.dimensions[1]}` : null}</Row>
        <Row label="Position">{node.is_root ? 'Origin of the tree' : `Step ${node.depth} from origin`}</Row>
        <Row label="Copied from">{parent ? (parent.kind === 'upload' ? 'Your image' : platformName(parent.version.platform)) : null}</Row>
        <Row label="Change vs. parent">{node.diff_fraction != null ? `${pct(node.diff_fraction, 1)} of the frame` : null}</Row>
      </div>
      {fa?.visual_summary && fa.source === 'gemini' && (
        <div className="panel-note"><strong>Gemini:</strong> {fa.visual_summary}</div>
      )}
      <div className="panel-links">
        {isHttp(v.url) && !isUpload && <a className="btn btn-secondary" href={v.url} target="_blank" rel="noopener noreferrer">Open page <ExternalLink size={13} /></a>}
        {isHttp(v.source_image_url) && <a className="btn btn-secondary" href={v.source_image_url} target="_blank" rel="noopener noreferrer">Open image <ExternalLink size={13} /></a>}
      </div>
    </>
  );
}

function EdgeDetail({ edge, tree }) {
  const src = tree.nodes[edge.source_id];
  const tgt = tree.nodes[edge.target_id];
  const g = edge.gemini_evidence;
  const name = (n) => (n?.kind === 'upload' ? 'Your image' : platformName(n?.version?.platform));
  const factors = [
    ['Visual match', edge.visual_score],
    ['Time order', edge.time_score],
    ['Edit direction', edge.edit_direction_score],
    ['Metadata', edge.metadata_score],
    ['Source trust', edge.trust_score],
  ];
  return (
    <>
      <div className="panel-pair">
        {[src, tgt].map((n, i) => (
          <div key={i} className="panel-pair-item">
            <div className="panel-pair-img">{n?.version?.image_url && <img src={n.version.image_url} alt="" />}</div>
            <span>{i === 0 ? 'From' : 'To'} · {name(n)}</span>
          </div>
        ))}
      </div>
      <div className="panel-title"><h2>{name(src)} → {name(tgt)}</h2></div>
      {g?.visual_diff_summary && <p className="panel-caption">{g.visual_diff_summary}</p>}
      {g?.tampering_detected && g.tampering_details && <div className="alert alert-danger">{g.tampering_details}</div>}
      <h3 className="section-title">Link confidence · {pct(edge.total_score)}</h3>
      <div className="factors">
        {factors.map(([label, val]) => (
          <div key={label} className="factor">
            <span>{label}</span>
            <div className="meter"><div className="meter-fill is-accent" style={{ width: pct(val || 0) }} /></div>
            <span className="factor-val">{pct(val || 0)}</span>
          </div>
        ))}
      </div>
    </>
  );
}

export default function DetailPanel({ selection, tree, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!selection) return null;
  return (
    <div className="overlay" onClick={onClose}>
      <aside className="panel" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true">
        <button className="icon-btn panel-close" onClick={onClose} aria-label="Close"><X size={18} /></button>
        {selection.type === 'node'
          ? <NodeDetail node={selection.item} tree={tree} />
          : <EdgeDetail edge={selection.item} tree={tree} />}
      </aside>
    </div>
  );
}
