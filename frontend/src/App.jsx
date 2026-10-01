import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ArrowRight, ChevronRight, Copy, Check, ImagePlus, Search, GitBranch, Clock, Images,
  ScanSearch, FileSearch, Crop, Cpu, ShieldCheck, Activity, Plus, X,
} from 'lucide-react';
import './styles.css';
import { startInvestigation, pollInvestigation, fetchHealth } from './api.js';
import PipelineRunner from './components/PipelineRunner.jsx';
import LineageTree from './components/LineageTree.jsx';
import Timeline from './components/Timeline.jsx';
import SimilarImages from './components/SimilarImages.jsx';
import ForensicVerdict from './components/ForensicVerdict.jsx';
import DetailPanel from './components/DetailPanel.jsx';
import Proof from './components/Proof.jsx';
import Reveal from './components/Reveal.jsx';
import { formatDate, platformName, pct } from './lib.js';

const TABS = [
  { id: 'proof', label: 'Proof', icon: ShieldCheck },
  { id: 'tree', label: 'Lineage', icon: GitBranch },
  { id: 'timeline', label: 'Timeline', icon: Clock },
  { id: 'similar', label: 'Copies', icon: Images },
  { id: 'signals', label: 'Signals', icon: Activity },
];

const METHODS = [
  { icon: Search, title: 'Copies online', text: 'Reverse-searches Yandex and Bing, then aligns every hit pixel by pixel to outline exactly what changed.' },
  { icon: Crop, title: 'Pasted regions', text: 'Faces and edited areas are searched on their own, so a head or object cut from another photo is traced to its source.' },
  { icon: FileSearch, title: 'Camera preview', text: 'Cameras hide a preview inside the file. If an editor left it behind, it shows the picture before it was changed.' },
  { icon: Cpu, title: 'Forgery model', text: 'TruFor, a CVPR 2023 detector, reads the camera-noise fingerprint and localises spliced or retouched areas.' },
  { icon: GitBranch, title: 'Lineage', text: 'Copies are dated and ordered into a family tree from the earliest version to yours.' },
  { icon: ScanSearch, title: 'Proof, not a score', text: 'Every verdict lists the evidence behind it, from direct proof down to supporting signals.' },
];

const FAQ = [
  {
    q: 'What counts as proof that an image was edited?',
    a: 'A reference that shows different content: an earlier copy online that differs, the camera preview stored in the file, or a region found pixel-for-pixel in a different photo. Detector scores are shown as evidence, but labelled as such.',
  },
  {
    q: 'What if the original was never published?',
    a: 'Then the unedited pixels cannot be recovered. TraceLens instead proves the edit: it localises the changed area, checks the embedded camera preview, and traces any pasted-in region to where it came from.',
  },
  {
    q: 'Does an "original" verdict mean the image is authentic?',
    a: 'It means no evidence of editing was found. Forensic checks can miss careful edits, so absence of evidence is not proof of authenticity.',
  },
  {
    q: 'Where are my images processed?',
    a: 'On the TraceLens server. To search the web, a downscaled copy is sent to the Yandex and Bing reverse image search engines, and to Gemini when its quota is available.',
  },
];

function Status({ health }) {
  const state = !health ? 'pending' : health.status !== 'healthy' ? 'down' : 'ok';
  const text = { pending: 'Connecting', down: 'Backend offline', ok: 'Online' }[state];
  return <span className={`status status-${state}`}><span className="status-dot" />{text}</span>;
}

function UploadCard({ onSubmit }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState('');
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  const pick = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };
  const reset = () => {
    setFile(null);
    setPreview(null);
    setCaption('');
    if (inputRef.current) inputRef.current.value = '';
  };
  const submit = () => {
    if (!file) return;
    onSubmit(file, caption, preview);
    reset();
  };

  return (
    <div className="upload-card">
      <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files?.[0])} />
      {!file ? (
        <button
          className={`dropzone ${drag ? 'is-drag' : ''}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); }}
        >
          <span className="icon-tile"><ImagePlus size={18} /></span>
          <span className="dropzone-title">Drop an image to analyze</span>
          <span className="dropzone-sub">or click to browse · JPG, PNG, WebP</span>
        </button>
      ) : (
        <div className="selected">
          <div className="selected-frame"><img src={preview} alt="Selected" /></div>
          <div className="selected-row">
            <div className="selected-meta">
              <div className="selected-name">{file.name}</div>
              <div className="selected-size">{(file.size / 1024).toFixed(0)} KB</div>
            </div>
            <button className="icon-btn" onClick={reset} aria-label="Remove"><X size={16} /></button>
          </div>
          <input
            className="input"
            placeholder="Claim or caption that came with it (optional)"
            value={caption}
            onChange={(e) => setCaption(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submit()}
          />
          <button className="btn btn-primary btn-block" onClick={submit}>Analyze image <ArrowRight size={15} /></button>
        </div>
      )}
    </div>
  );
}

function FaqItem({ q, a, open, onToggle }) {
  return (
    <div className={`faq-item ${open ? 'is-open' : ''}`}>
      <button className="faq-q" onClick={onToggle} aria-expanded={open}>
        <span>{q}</span>
        {open ? <X size={14} /> : <Plus size={14} />}
      </button>
      <div className="faq-a"><p>{a}</p></div>
    </div>
  );
}

function Landing({ onSubmit }) {
  const [openFaq, setOpenFaq] = useState(0);
  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <span className="badge">Image forensics <ChevronRight size={13} /></span>
          <h1>
            Prove an image
            <span className="dim"> was edited</span>
          </h1>
          <p className="lede">
            Upload a picture. TraceLens finds its copies online, outlines what changed, traces
            pasted-in parts to their source, and shows the path from the original to your copy.
          </p>
          <div className="hero-points">
            <span>Copies online</span><span>Pasted regions</span><span>Camera preview</span><span>Forgery model</span>
          </div>
        </div>
        <div className="hero-upload">
          <UploadCard onSubmit={onSubmit} />
        </div>
      </section>

      <section className="section">
        <Reveal as="h2" className="section-heading">
          How it proves an edit <span className="dim">without the original</span>
        </Reveal>
        <div className="method-grid">
          {METHODS.map((m, i) => (
            <Reveal key={m.title} delay={i * 60} className="method">
              <span className="icon-tile"><m.icon size={16} /></span>
              <h3>{m.title}</h3>
              <p>{m.text}</p>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section section-narrow">
        <Reveal as="h2" className="section-heading center">Questions, <span className="dim">answered</span></Reveal>
        <Reveal className="faq">
          {FAQ.map((f, i) => (
            <FaqItem key={f.q} {...f} open={openFaq === i} onToggle={() => setOpenFaq(openFaq === i ? -1 : i)} />
          ))}
        </Reveal>
      </section>
    </>
  );
}

function Summary({ result, previewUrl, onCopy, copied }) {
  const c = result.classification || {};
  const tampered = c.verdict === 'TAMPERED';
  const decisive = c.proof_count?.decisive || 0;
  const root = result.nodes?.[result.root_id];
  const rootIsCopy = root && root.kind === 'web_copy';
  const uploadNode = result.nodes?.[result.upload_node_id];
  const stats = [
    { k: 'pieces of proof', v: (result.proof || []).filter((p) => p.strength !== 'supporting').length },
    { k: 'copies online', v: (result.web_matches || []).length },
    { k: 'steps from origin', v: rootIsCopy ? result.stage_count : '—' },
    { k: 'earliest copy', v: rootIsCopy ? (formatDate(root.version.timestamp) || platformName(root.version.platform)) : '—' },
  ];
  return (
    <section className="summary">
      <div className="summary-frame"><img src={uploadNode?.version?.image_url || previewUrl} alt="Analyzed" /></div>
      <div className="summary-body">
        <span className={`verdict-tag ${tampered ? 'is-edited' : 'is-clean'}`}>
          <span className="verdict-dot" />
          {tampered ? 'Edited' : 'No edit found'} · {pct(c.confidence || 0)} confidence
        </span>
        <h1 className="summary-heading">
          {tampered
            ? <>This image was edited.<span className="dim">{decisive ? ` ${decisive} piece${decisive > 1 ? 's' : ''} of direct proof.` : ' Based on detector evidence.'}</span></>
            : <>No edit was found.<span className="dim"> That is not the same as proof it is authentic.</span></>}
        </h1>
        <p className="summary-text">{result.verdict_explanation}</p>
        <div className="summary-actions">
          <button className="btn btn-ghost btn-sm" onClick={onCopy}>
            {copied ? <Check size={14} /> : <Copy size={14} />}{copied ? 'Copied' : 'Copy report'}
          </button>
        </div>
      </div>
      <dl className="stats">
        {stats.map((s) => (
          <div key={s.k} className="stat"><dd>{s.v}</dd><dt>{s.k}</dt></div>
        ))}
      </dl>
    </section>
  );
}

function buildReport(r) {
  const c = r.classification || {};
  const lines = [
    '# TraceLens report',
    `Verdict: ${c.verdict === 'TAMPERED' ? 'EDITED' : 'NO EDIT FOUND'} (${pct(c.confidence || 0)} confidence)`,
    '',
    r.verdict_explanation || '',
    '',
    '## Proof',
    ...(r.proof || []).map((p) => `- [${p.strength}] ${p.title}: ${p.detail}${p.links?.length ? ` (${p.links.map((l) => l.url).join(', ')})` : ''}`),
    '',
    '## Path from origin',
    ...(r.lineage_path || []).map((id, i) => {
      const n = r.nodes[id];
      const name = n.kind === 'upload' ? 'Your image' : platformName(n.version.platform);
      const date = n.version.timestamp && n.kind !== 'upload' ? ` (${formatDate(n.version.timestamp)})` : '';
      const url = n.version.url && n.kind !== 'upload' ? ` — ${n.version.url}` : '';
      return `${i}. ${name}${date}${url}`;
    }),
    '',
    '## Closest copies online',
    ...(r.web_matches || []).map((m) => `- ${platformName(m.platform)}: ${m.page_url}`),
  ];
  return lines.join('\n');
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [job, setJob] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [tab, setTab] = useState('proof');
  const [selection, setSelection] = useState(null);
  const [copied, setCopied] = useState(false);
  const cancelRef = useRef(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth({ status: 'offline' }));
    return () => cancelRef.current?.();
  }, []);

  const analyze = async (file, caption, preview) => {
    cancelRef.current?.();
    setResult(null);
    setSelection(null);
    setPreviewUrl(preview);
    setJob({ status: 'queued', filename: file.name, stages: [] });
    window.scrollTo({ top: 0, behavior: 'smooth' });
    try {
      const started = await startInvestigation(file, { caption });
      setJob(started);
      cancelRef.current = pollInvestigation(
        started.id,
        (snap) => {
          setJob(snap);
          if (snap.status === 'done' && snap.result) {
            setResult(snap.result);
            setTab(snap.result.classification?.verdict === 'TAMPERED' ? 'proof' : 'tree');
          }
        },
        (err) => setJob((j) => ({ ...j, status: 'failed', error: `Lost contact with the server: ${err.message}` })),
      );
    } catch (err) {
      setJob((j) => ({ ...j, status: 'failed', error: `${err.message}. Is the backend running on port 8000?` }));
    }
  };

  const startOver = () => {
    cancelRef.current?.();
    setJob(null);
    setResult(null);
    setSelection(null);
  };

  const copyReport = () => {
    navigator.clipboard.writeText(buildReport(result)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  };

  const selectNode = useCallback((node) => setSelection({ type: 'node', item: node }), []);
  const selectEdge = useCallback((edge) => setSelection({ type: 'edge', item: edge }), []);
  const running = job && (job.status === 'queued' || job.status === 'running');
  const counts = result && {
    proof: (result.proof || []).length,
    similar: (result.web_matches || []).length,
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-inner">
          <button className="brand" onClick={startOver} aria-label="TraceLens home">
            <span className="brand-mark"><GitBranch size={14} strokeWidth={2.4} /></span>
            tracelens
          </button>
          <div className="topbar-right">
            <Status health={health} />
            {job && !running && (
              <button className="btn btn-primary btn-sm" onClick={startOver}>New analysis <ArrowRight size={14} /></button>
            )}
          </div>
        </div>
      </header>

      <main className="container">
        {!job && <Landing onSubmit={analyze} />}

        {running && (
          <section className="running">
            <div className="running-preview">
              <div className="running-frame">
                {previewUrl && <img src={previewUrl} alt="Uploaded" />}
                <span className="scanline" />
              </div>
              <div className="running-name">{job.filename}</div>
              {job.classification?.stage === 'verdict' && (
                <span className={`verdict-tag ${job.classification.verdict === 'TAMPERED' ? 'is-edited' : 'is-clean'}`}>
                  <span className="verdict-dot" />
                  {job.classification.verdict === 'TAMPERED' ? 'Edited' : 'No edit found'} · {pct(job.classification.confidence)}
                </span>
              )}
            </div>
            <PipelineRunner job={job} />
          </section>
        )}

        {job?.status === 'failed' && (
          <section className="stack">
            <PipelineRunner job={job} />
            <div><button className="btn btn-primary" onClick={startOver}>Try another image</button></div>
          </section>
        )}

        {result && (
          <section className="stack">
            <Summary result={result} previewUrl={previewUrl} onCopy={copyReport} copied={copied} />
            <PipelineRunner job={job} defaultOpen={false} />

            <div className="results">
              <nav className="tabs" role="tablist">
                {TABS.map((t) => (
                  <button key={t.id} role="tab" aria-selected={tab === t.id} className={`tab ${tab === t.id ? 'is-active' : ''}`} onClick={() => setTab(t.id)}>
                    <span className="tab-tile">
                      <t.icon size={17} />
                      {counts[t.id] > 0 && <span className="tab-count">{counts[t.id]}</span>}
                    </span>
                    <span className="tab-label">{t.label}</span>
                  </button>
                ))}
              </nav>
              <div className="tab-body" key={tab}>
                {tab === 'proof' && <Proof proof={result.proof} verdict={result.classification?.verdict} />}
                {tab === 'tree' && (
                  Object.keys(result.nodes || {}).length > 1
                    ? <LineageTree tree={result} onSelect={selectNode} onSelectEdge={selectEdge} />
                    : <div className="empty">No copies of this image were found online, so there is no lineage to draw.</div>
                )}
                {tab === 'timeline' && <Timeline tree={result} onSelect={selectNode} />}
                {tab === 'similar' && (
                  <SimilarImages matches={result.web_matches || []} platforms={result.platforms} search={result.search_summary} />
                )}
                {tab === 'signals' && (
                  <ForensicVerdict classification={result.classification} uploadUrl={result.nodes?.[result.upload_node_id]?.version?.image_url} />
                )}
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <div className="footer-inner">
          <span>tracelens</span>
          <span className="dim">Forensic evidence, not certainty. Uses TruFor (GRIP-UNINA) under its non-profit licence.</span>
        </div>
      </footer>

      {result && <DetailPanel selection={selection} tree={result} onClose={() => setSelection(null)} />}
    </div>
  );
}
