import React, { useMemo } from 'react';
import { HelpCircle, ArrowRight } from 'lucide-react';
import PlatformIcon from './PlatformIcon.jsx';
import { formatDate, platformName, pct } from '../lib.js';

function TreeNode({ node, stage, isTampered, onSelect, depth = 0 }) {
  const v = node.version;
  const isUpload = node.kind === 'upload';
  const isInferred = node.kind === 'inferred_original';
  const edited = node.altered_here && node.diff_fraction != null;

  let tag = null;
  if (node.is_root) tag = <span className="tag tag-origin">Origin</span>;
  else if (isUpload) tag = <span className={`tag ${isTampered ? 'tag-danger' : 'tag-accent'}`}>Your image</span>;
  else if (stage != null) tag = <span className="tag">Stage {stage}</span>;

  return (
    <button
      className={`tnode ${stage != null ? 'on-path' : ''} ${isUpload ? 'is-upload' : ''} ${isUpload && isTampered ? 'is-tampered' : ''} ${node.is_root ? 'is-root' : ''}`}
      onClick={() => onSelect?.(node)}
      style={{ animationDelay: `${depth * 0.14}s` }}
    >
      <div className="tnode-img">
        {isInferred ? (
          <div className="tnode-placeholder"><HelpCircle size={22} /><span>Original not found online</span></div>
        ) : (
          v.image_url && <img src={v.image_url} alt="" loading="lazy" onError={(e) => { e.currentTarget.style.visibility = 'hidden'; }} />
        )}
        {tag && <div className="tnode-tag">{tag}</div>}
      </div>
      <div className="tnode-body">
        <div className="tnode-platform">
          <PlatformIcon url={v.url} isUpload={isUpload} size={14} />
          <span>{isUpload ? 'Uploaded image' : isInferred ? 'Unknown source' : platformName(v.platform)}</span>
        </div>
        <div className="tnode-meta">
          <span>{isUpload ? 'Today' : formatDate(v.timestamp) || 'Date unknown'}</span>
          {edited && <span className={isUpload && !isTampered ? 'muted' : 'danger'}>{isUpload && !isTampered ? 'Differs' : 'Edited'} {pct(node.diff_fraction, 1)}</span>}
        </div>
      </div>
    </button>
  );
}

function Branch({ id, ctx, depth = 0 }) {
  const { nodes, childrenOf, edgeByTarget, stageOf, onSelect, isTampered } = ctx;
  const node = nodes[id];
  if (!node || depth > 30) return null;
  const kids = childrenOf[id] || [];
  return (
    <div className="branch">
      <TreeNode node={node} stage={stageOf[id]} isTampered={isTampered} onSelect={onSelect} depth={depth} />
      {kids.length > 0 && (
        <div className={`branch-children ${kids.length === 1 ? 'single' : ''}`}>
          {kids.map((cid) => {
            const child = nodes[cid];
            const edge = edgeByTarget[cid];
            const onPath = stageOf[cid] != null && stageOf[id] != null;
            const edited = child?.altered_here && child?.diff_fraction != null && (child.kind !== 'upload' || isTampered);
            return (
              <div key={cid} className={`branch-child ${onPath ? 'on-path' : ''}`}>
                <button style={{ animationDelay: `${depth * 0.14 + 0.07}s` }} className={`edge-label ${edited ? 'is-edited' : ''}`} onClick={() => edge && ctx.onSelectEdge?.(edge)} title="Inspect this step">
                  {edited ? `Edited · ${pct(child.diff_fraction, 1)}` : edge ? `${Math.round(edge.total_score * 100)}% match` : 'copy'}
                </button>
                <Branch id={cid} ctx={ctx} depth={depth + 1} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function LineageTree({ tree, onSelect, onSelectEdge }) {
  const nodes = tree.nodes || {};
  const path = tree.lineage_path || [];
  const isTampered = tree.classification?.verdict === 'TAMPERED';

  const ctx = useMemo(() => {
    const childrenOf = {};
    const edgeByTarget = {};
    (tree.edges || []).forEach((e) => { edgeByTarget[e.target_id] = e; });
    const onPath = new Set(path);
    Object.values(nodes).forEach((n) => {
      if (n.parent_id && nodes[n.parent_id]) (childrenOf[n.parent_id] ||= []).push(n.version.id);
    });
    // Keep the branch leading to the upload in the centre-left so the main story reads first
    Object.values(childrenOf).forEach((list) => list.sort((a, b) => Number(onPath.has(b)) - Number(onPath.has(a))));
    const stageOf = {};
    path.forEach((id, i) => { stageOf[id] = i; });
    return { nodes, childrenOf, edgeByTarget, stageOf, onSelect, onSelectEdge, isTampered };
  }, [tree, nodes, path, onSelect, onSelectEdge, isTampered]);

  const stageCount = tree.stage_count ?? Math.max(0, path.length - 1);
  const firstEdited = path.map((id) => nodes[id]).find((n) => n?.altered_here && (n.kind !== 'upload' || isTampered));

  return (
    <div className="lineage">
      {path.length > 1 && (
        <div className="path-summary">
          <div className="path-steps">
            {path.map((id, i) => {
              const n = nodes[id];
              if (!n) return null;
              const label = n.kind === 'upload' ? 'Your image' : n.kind === 'inferred_original' ? 'Original' : platformName(n.version.platform);
              return (
                <React.Fragment key={id}>
                  {i > 0 && <ArrowRight size={14} className="path-arrow" />}
                  <button className={`path-chip ${n.altered_here && (n.kind !== 'upload' || isTampered) ? 'is-edited' : ''}`} onClick={() => onSelect?.(n)}>
                    {n.kind !== 'inferred_original' && <PlatformIcon url={n.version.url} isUpload={n.kind === 'upload'} size={14} />}
                    {label}
                  </button>
                </React.Fragment>
              );
            })}
          </div>
          <div className="path-meta">
            <strong>{stageCount}</strong> {stageCount === 1 ? 'step' : 'steps'} from the origin to your image
            {firstEdited && <> · first edited at <strong className="danger">{firstEdited.kind === 'upload' ? 'your image' : platformName(firstEdited.version.platform)}</strong></>}
          </div>
        </div>
      )}

      <div className="tree-scroll">
        <div className="tree">
          <Branch id={tree.root_id} ctx={ctx} />
        </div>
      </div>
    </div>
  );
}
