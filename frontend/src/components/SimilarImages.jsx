import React from 'react';
import { ExternalLink } from 'lucide-react';
import PlatformIcon from './PlatformIcon.jsx';
import { formatDate, isHttp, platformName, pct } from '../lib.js';

export default function SimilarImages({ matches, platforms, search }) {
  if (!matches.length) {
    return (
      <div className="empty">
        No verified copies of this image were found online.
        {search?.yandex_url && <> You can review the raw results on <a href={search.yandex_url} target="_blank" rel="noopener noreferrer">Yandex</a>.</>}
      </div>
    );
  }

  return (
    <div className="similar">
      {platforms?.length > 0 && (
        <div className="found-on">
          <span className="found-on-label">Found on</span>
          {platforms.map((p) => (
            <a key={p.platform} className="chip" href={p.urls?.[0]} target="_blank" rel="noopener noreferrer" title={p.urls?.[0]}>
              <PlatformIcon url={p.urls?.[0]} size={14} />
              {platformName(p.platform)}
              {p.count > 1 && <span className="chip-count">{p.count}</span>}
            </a>
          ))}
        </div>
      )}

      <div className="sim-grid">
        {matches.map((m) => {
          const differs = (m.diff_fraction || 0) >= 0.01;
          return (
            <article key={m.match_id} className="sim-card">
              <a className="sim-img" href={isHttp(m.page_url) ? m.page_url : m.thumbnail_url} target="_blank" rel="noopener noreferrer">
                {(m.thumbnail_url || m.image_url) && <img src={m.thumbnail_url || m.image_url} alt={m.page_title || ''} loading="lazy" />}
                <span className="sim-score">{pct(m.similarity_score)} match</span>
              </a>
              <div className="sim-body">
                <div className="sim-platform">
                  <PlatformIcon url={m.page_url} size={14} />
                  <span>{platformName(m.platform)}</span>
                  <span className={`tag ${differs ? 'tag-danger' : 'tag-success'}`}>{differs ? `Differs ${pct(m.diff_fraction, 1)}` : 'Same image'}</span>
                </div>
                <div className="sim-title" title={m.page_title || ''}>{m.page_title || 'Untitled page'}</div>
                <div className="sim-meta">
                  {formatDate(m.published_at) && <span>{formatDate(m.published_at)}</span>}
                  {m.dimensions && <span>{m.dimensions[0]}×{m.dimensions[1]}</span>}
                </div>
                <div className="sim-links">
                  {isHttp(m.page_url) && <a href={m.page_url} target="_blank" rel="noopener noreferrer">Page <ExternalLink size={11} /></a>}
                  {isHttp(m.source_image_url) && <a href={m.source_image_url} target="_blank" rel="noopener noreferrer">Image <ExternalLink size={11} /></a>}
                </div>
              </div>
            </article>
          );
        })}
      </div>

      {(search?.yandex_url || search?.search_url) && (
        <p className="footnote">
          Showing the {matches.length} closest verified copies.{' '}
          {search.yandex_url && <a href={search.yandex_url} target="_blank" rel="noopener noreferrer">All Yandex results</a>}
          {search.yandex_url && search.search_url && ' · '}
          {search.search_url && <a href={search.search_url} target="_blank" rel="noopener noreferrer">All Bing results</a>}
        </p>
      )}
    </div>
  );
}
