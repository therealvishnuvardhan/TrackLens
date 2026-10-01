import React, { useState } from 'react';
import { Globe, User } from 'lucide-react';
import { faviconUrl } from '../lib.js';

export default function PlatformIcon({ url, isUpload = false, size = 16 }) {
  const [failed, setFailed] = useState(false);
  const src = faviconUrl(url);
  if (isUpload) {
    return <span className="pi pi-upload" style={{ width: size, height: size }}><User size={size * 0.7} /></span>;
  }
  if (!src || failed) {
    return <span className="pi" style={{ width: size, height: size }}><Globe size={size * 0.75} /></span>;
  }
  return (
    <img className="pi pi-img" src={src} alt="" width={size} height={size} onError={() => setFailed(true)} loading="lazy" />
  );
}
