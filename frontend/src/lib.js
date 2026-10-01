export const CATEGORY_LABEL = {
  upload: 'Your image',
  social: 'Social media',
  news: 'News',
  fact_check: 'Fact-check',
  reference: 'Reference',
  stock: 'Stock photo',
  photo: 'Photo sharing',
  blog: 'Blog',
  web: 'Website',
  inferred: 'Not found online',
};

export const isHttp = (u) => typeof u === 'string' && /^https?:\/\//i.test(u);

export function formatDate(iso, { withYear = true } = {}) {
  if (!iso) return null;
  const d = new Date(iso);
  if (isNaN(d)) return null;
  return d.toLocaleDateString('en-US', withYear
    ? { day: 'numeric', month: 'short', year: 'numeric' }
    : { day: 'numeric', month: 'short' });
}

export function domainOf(url) {
  if (!isHttp(url)) return null;
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return null;
  }
}

export function faviconUrl(url) {
  const d = domainOf(url);
  return d ? `https://www.google.com/s2/favicons?domain=${d}&sz=64` : null;
}

// "Website: example.com" / "News: bbc.co.uk" -> "example.com" / "bbc.co.uk"
export function platformName(platform) {
  if (!platform) return 'Unknown';
  const m = platform.match(/^(?:Website|News|Fact-check|Institution|Photographer|Wallpaper\/meme site):\s*(.+)$/);
  return m ? m[1] : platform;
}

export const pct = (x, digits = 0) => `${(x * 100).toFixed(digits)}%`;

export function durationLabel(seconds) {
  if (seconds == null) return '';
  if (seconds < 1) return '<1s';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}
