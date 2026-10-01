from __future__ import annotations
import io
import re
import json
import base64
import html as html_lib
import urllib.parse
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from backend.search.page_scraper import BROWSER_UA, PageScraper

BING_BASE = "https://www.bing.com"
YANDEX_BASE = "https://yandex.com"
_ITEM_RE = re.compile(r'class="iusc"[^>]*?\sm="([^"]+)"')
_BLOCKED_WORDS = ("porn", "xxx", "sex", "adult", "nsfw", "escort", "hentai", "realno.tv")
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid")


def _clean_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return url
    if url.startswith("//"):
        url = "https:" + url
    try:
        parts = urllib.parse.urlsplit(url)
        query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
                 if not k.startswith(_TRACKING_PARAMS)]
        return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))
    except Exception:
        return url


def _blocked(url: Optional[str], title: Optional[str]) -> bool:
    text = f"{url or ''} {title or ''}".lower()
    return any(w in text for w in _BLOCKED_WORDS)


class ReverseImageSearch:
    """
    Live reverse image search (no API key required).

    1. Yandex "search by image" -> "Sites containing the image": pages that host copies
       of this exact picture (including crops/resizes), with the full-size image URL.
    2. Bing Visual Search: visually matching images and Bing's best guess of the subject.
    3. A keyword image search (Gemini's description or Bing's guess) to widen the
       candidate pool when the engines above return little.

    Every result carries real, clickable URLs: the hosting page, the full-size image,
    and a thumbnail. Results still need visual verification (see MatchVerifier)
    because "visually similar" also includes unrelated look-alikes.
    """

    def __init__(self, timeout: float = 20.0):
        self.timeout = timeout

    @staticmethod
    def _session() -> requests.Session:
        s = requests.Session()
        s.headers.update({"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.8"})
        return s

    def search(self, image_path: str, query_hint: Optional[str] = None, max_results: int = 45,
               use_keyword: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "engines": [],
            "best_query": None,
            "search_url": None,
            "yandex_url": None,
            "keyword_query": None,
            "keyword_search_url": None,
            "results": [],
            "errors": [],
        }
        seen = set()
        jpeg = self._jpeg_bytes(image_path)

        def add(results: List[Dict[str, Any]], source: str):
            for r in results:
                if _blocked(r.get("page_url"), r.get("title")):
                    continue
                key = r.get("image_url") or r.get("page_url")
                if key and key not in seen:
                    seen.add(key)
                    r["source"] = source
                    out["results"].append(r)

        try:
            yandex = self._yandex_sites(self._session(), jpeg)
            out["yandex_url"] = yandex["search_url"]
            add(yandex["results"], "yandex_sites")
            out["engines"].append("Yandex reverse image search")
        except Exception as e:
            out["errors"].append(f"Yandex reverse search failed: {e}")

        bing = self._session()
        try:
            try:
                visual = self._bing_visual(bing, jpeg)
            except requests.HTTPError:
                # Bing rejects some sizes/aspect ratios; retry with a smaller, square-padded copy
                visual = self._bing_visual(bing, self._jpeg_bytes(image_path, max_side=600, pad_square=True))
            out["best_query"] = visual["best_query"]
            out["search_url"] = visual["search_url"]
            add(visual["results"], "bing_visual")
            out["engines"].append("Bing visual search")
        except Exception as e:
            out["errors"].append(f"Bing visual search failed: {e}")

        query = (query_hint or "").strip() or out["best_query"]
        if use_keyword and query and len(out["results"]) < 25:
            try:
                kw = self._bing_keyword(bing, query)
                out["keyword_query"] = query
                out["keyword_search_url"] = kw["search_url"]
                add(kw["results"], "bing_keyword")
                out["engines"].append("Bing keyword image search")
            except Exception as e:
                out["errors"].append(f"Bing keyword search failed: {e}")

        out["results"] = out["results"][:max_results]
        return out

    @staticmethod
    def _jpeg_bytes(image_path: str, max_side: int = 1000, pad_square: bool = False) -> bytes:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            if pad_square:
                side = max(im.size)
                canvas = Image.new("RGB", (side, side), (0, 0, 0))
                canvas.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2))
                im = canvas
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
        return buf.getvalue()

    # ------------------------------------------------------------------ Yandex

    def _yandex_sites(self, session: requests.Session, jpeg: bytes) -> Dict[str, Any]:
        request = json.dumps({"blocks": [{"block": "b-page_type_search-by-image__link"}]})
        resp = session.post(
            f"{YANDEX_BASE}/images/search?rpt=imageview&format=json&request={urllib.parse.quote(request)}",
            files={"upfile": ("blob", jpeg, "image/jpeg")},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        params = resp.json()["blocks"][0]["params"]
        cbir_id, original = params["cbirId"], params["originalImageUrl"]
        search_url = (
            f"{YANDEX_BASE}/images/search?cbir_id={urllib.parse.quote(cbir_id)}&rpt=imageview"
            f"&url={urllib.parse.quote(original)}&cbir_page=sites"
        )
        page = session.get(search_url, timeout=self.timeout)
        page.raise_for_status()
        if "showcaptcha" in page.url or "CbirSites" not in page.text and "captcha" in page.text.lower():
            raise RuntimeError("blocked by captcha")
        return {"search_url": search_url, "results": self._parse_yandex_sites(page.text)}

    @staticmethod
    def _parse_yandex_sites(page_html: str) -> List[Dict[str, Any]]:
        results = []
        for block in page_html.split('<li class="CbirSites-Item">')[1:]:
            image = re.search(r'CbirSites-ItemThumb.*?<a href="([^"]+)"', block, re.S)
            thumb = re.search(r'<img class="Image Thumb-Image" src="([^"]+)"', block)
            dims = re.search(r'Thumb-Mark[^>]*>(\d+)\D(\d+)<', block)
            title = re.search(r'CbirSites-ItemTitle"><a href="([^"]+)"[^>]*>([^<]*)<', block)
            desc = re.search(r'CbirSites-ItemDescription[^>]*>([^<]*)<', block)
            if not title:
                continue
            page_url = _clean_url(html_lib.unescape(title.group(1)))
            platform, category, trust = PageScraper.classify_url(page_url)
            results.append({
                "title": html_lib.unescape(title.group(2)).strip() or None,
                "description": html_lib.unescape(desc.group(1)).strip() if desc else None,
                "page_url": page_url,
                "image_url": _clean_url(html_lib.unescape(image.group(1))) if image else None,
                "thumb_url": _clean_url(html_lib.unescape(thumb.group(1))) if thumb else None,
                "reported_dimensions": [int(dims.group(1)), int(dims.group(2))] if dims else None,
                "domain": PageScraper.domain_of(page_url),
                "platform": platform,
                "category": category,
                "trust": trust,
            })
        return results

    # ------------------------------------------------------------------ Bing

    def _bing_visual(self, session: requests.Session, jpeg: bytes) -> Dict[str, Any]:
        b64 = base64.b64encode(jpeg).decode("ascii")
        resp = session.post(
            f"{BING_BASE}/images/kblob",
            files={"imageBin": (None, b64), "cbir": (None, "sbi")},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        redirect = resp.json().get("redirectUrl")
        if not redirect:
            raise RuntimeError("no redirect from upload endpoint")

        search_url = f"{BING_BASE}/images{redirect}"
        params = urllib.parse.parse_qs(urllib.parse.urlparse(redirect).query)
        best_query = (params.get("q") or [None])[0]

        page = session.get(search_url, timeout=self.timeout)
        page.raise_for_status()
        return {
            "best_query": best_query,
            "search_url": search_url,
            "results": self._parse_bing(page.text),
        }

    def _bing_keyword(self, session: requests.Session, query: str) -> Dict[str, Any]:
        search_url = f"{BING_BASE}/images/search?q={urllib.parse.quote_plus(query)}&form=HDRSC2&first=1"
        page = session.get(search_url, timeout=self.timeout)
        page.raise_for_status()
        return {"search_url": search_url, "results": self._parse_bing(page.text)}

    @staticmethod
    def _parse_bing(page_html: str) -> List[Dict[str, Any]]:
        results = []
        for raw in _ITEM_RE.findall(page_html):
            try:
                m = json.loads(html_lib.unescape(raw))
            except Exception:
                continue
            page_url = _clean_url(m.get("purl"))
            image_url = m.get("murl")
            if not page_url and not image_url:
                continue
            title = (m.get("t") or m.get("desc") or "").replace("", "").replace("", "").strip()
            platform, category, trust = PageScraper.classify_url(page_url or image_url)
            results.append({
                "title": html_lib.unescape(title) or None,
                "description": None,
                "page_url": page_url,
                "image_url": image_url,
                "thumb_url": m.get("turl"),
                "reported_dimensions": None,
                "domain": PageScraper.domain_of(page_url or image_url),
                "platform": platform,
                "category": category,
                "trust": trust,
            })
        return results
