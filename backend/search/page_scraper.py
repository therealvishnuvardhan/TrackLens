from __future__ import annotations
import re
import html as html_lib
import urllib.parse
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import requests

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)


class PageScraper:
    """
    Scrapes web page metadata for reverse image search matches:
    1. Extracts publication timestamp (JSON-LD, OpenGraph, microdata, HTTP Last-Modified).
    2. Identifies source platform, category and baseline trust score.
    3. Extracts surrounding caption, headline, or alt-text for claim drift analysis.
    """

    # domain fragment -> (platform name, category, trust)
    PLATFORM_DOMAINS = {
        "x.com": ("X/Twitter", "social", 0.65),
        "twitter.com": ("X/Twitter", "social", 0.65),
        "twimg.com": ("X/Twitter", "social", 0.65),
        "instagram.com": ("Instagram", "social", 0.60),
        "reddit.com": ("Reddit", "social", 0.40),
        "redd.it": ("Reddit", "social", 0.40),
        "whatsapp.com": ("WhatsApp", "social", 0.30),
        "facebook.com": ("Facebook", "social", 0.45),
        "fbcdn.net": ("Facebook", "social", 0.45),
        "tiktok.com": ("TikTok", "social", 0.40),
        "telegram.org": ("Telegram", "social", 0.35),
        "t.me": ("Telegram", "social", 0.35),
        "youtube.com": ("YouTube", "social", 0.60),
        "pinterest.": ("Pinterest", "social", 0.35),
        "pinimg.com": ("Pinterest", "social", 0.35),
        "tumblr.com": ("Tumblr", "social", 0.35),
        "threads.net": ("Threads", "social", 0.50),
        "bsky.app": ("Bluesky", "social", 0.50),
        "linkedin.com": ("LinkedIn", "social", 0.55),
        "vk.com": ("VK", "social", 0.35),
        "weibo.com": ("Weibo", "social", 0.35),
        "quora.com": ("Quora", "social", 0.35),
        "9gag.com": ("9GAG", "social", 0.25),
        "imgur.com": ("Imgur", "social", 0.35),
        "flickr.com": ("Flickr", "photo", 0.75),
        "staticflickr.com": ("Flickr", "photo", 0.75),
        "wikipedia.org": ("Wikipedia", "reference", 0.90),
        "wikimedia.org": ("Wikimedia Commons", "reference", 0.92),
        "britannica.com": ("Britannica", "reference", 0.90),
        "loc.gov": ("Library of Congress", "reference", 0.97),
        "si.edu": ("Smithsonian", "reference", 0.95),
        "archive.org": ("Internet Archive", "reference", 0.85),
        "gettyimages.": ("Getty Images", "stock", 0.92),
        "alamy.com": ("Alamy", "stock", 0.85),
        "shutterstock.com": ("Shutterstock", "stock", 0.85),
        "istockphoto.com": ("iStock", "stock", 0.85),
        "adobe.com": ("Adobe Stock", "stock", 0.85),
        "dreamstime.com": ("Dreamstime", "stock", 0.75),
        "pixabay.com": ("Pixabay", "stock", 0.65),
        "unsplash.com": ("Unsplash", "stock", 0.75),
        "pexels.com": ("Pexels", "stock", 0.70),
        "snopes.com": ("Snopes", "fact_check", 0.95),
        "factcheck.org": ("FactCheck.org", "fact_check", 0.95),
        "politifact.com": ("PolitiFact", "fact_check", 0.95),
        "fullfact.org": ("Full Fact", "fact_check", 0.95),
        "leadstories.com": ("Lead Stories", "fact_check", 0.90),
        "factuel.afp.com": ("AFP Fact Check", "fact_check", 0.95),
        "factcheck.afp.com": ("AFP Fact Check", "fact_check", 0.95),
        "boomlive.in": ("BOOM Fact Check", "fact_check", 0.90),
        "altnews.in": ("Alt News", "fact_check", 0.90),
        "maldita.es": ("Maldita", "fact_check", 0.90),
        "newtral.es": ("Newtral", "fact_check", 0.90),
        "blogspot.com": ("Blogger", "blog", 0.40),
        "blogger.googleusercontent.com": ("Blogger", "blog", 0.40),
        "wordpress.com": ("WordPress", "blog", 0.40),
        "medium.com": ("Medium", "blog", 0.45),
        "livejournal.com": ("LiveJournal", "blog", 0.35),
        "kaskus.co.id": ("Kaskus forum", "social", 0.30),
        "artstation.com": ("ArtStation", "photo", 0.55),
        "deviantart.com": ("DeviantArt", "photo", 0.45),
        "play.google.com": ("Google Play", "web", 0.50),
        "wikidata.org": ("Wikidata", "reference", 0.90),
        "substack.com": ("Substack", "blog", 0.45),
    }

    NEWS_INDICATORS = [
        "news", "diario", "times", "post", "tribune", "gazette", "herald", "journal",
        "bbc", "cnn", "reuters", "apnews", "guardian", "eldiario", "elpais", "nytimes",
        "washingtonpost", "aljazeera", "nbc", "abc", "cbs", "foxnews", "npr", "dw.com",
        "france24", "lemonde", "telegraph", "independent", "express", "mirror", "dailymail",
        "hindustantimes", "ndtv", "thehindu", "indiatoday", "timesofindia", "indianexpress",
        "theatlantic", "time.com", "newsweek", "forbes", "bloomberg", "cnbc", "politico",
        "axios", "vox", "usatoday", "latimes", "chron", "sfgate", "breitbart", "globaltimes",
        "scmp", "straitstimes", "abc.net.au", "cbc.ca", "rte.ie", "rtve", "euronews", "skynews",
        "yahoo", "msn.com", "huffpost", "buzzfeed", "vice.com", "wired", "theverge", "nationalgeographic",
    ]

    DATE_PATTERNS = [
        re.compile(r'"datePublished"\s*:\s*"([^"]+)"', re.IGNORECASE),
        re.compile(r'"uploadDate"\s*:\s*"([^"]+)"', re.IGNORECASE),
        re.compile(r'"dateCreated"\s*:\s*"([^"]+)"', re.IGNORECASE),
        re.compile(r'<meta[^>]+(?:property|name|itemprop)=["\'](?:article:published_time|og:published_time|datePublished|pubdate|publishdate|date|dc\.date|DC\.date\.issued|sailthru\.date|parsely-pub-date)["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
        re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name|itemprop)=["\'](?:article:published_time|og:published_time|datePublished|pubdate|publishdate|date)["\']', re.IGNORECASE),
        re.compile(r'<time[^>]+datetime=["\']([^"\']+)["\']', re.IGNORECASE),
        re.compile(r'"dateModified"\s*:\s*"([^"]+)"', re.IGNORECASE),
        re.compile(r'<meta[^>]+property=["\'](?:article:modified_time|og:updated_time)["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    ]

    @classmethod
    def scrape_page(cls, url: str, html_override: Optional[str] = None, timeout: float = 6.0) -> Dict[str, Any]:
        """
        Visits the page and extracts platform, publication date, and caption.
        """
        platform, category, trust = cls.classify_url(url)

        res: Dict[str, Any] = {
            "url": url,
            "platform": platform,
            "category": category,
            "source_trust_score": trust,
            "published_at": None,
            "headline": None,
            "caption": None,
            "reachable": None,
        }

        html = html_override
        if not html and url.startswith("http"):
            try:
                resp = requests.get(
                    url,
                    timeout=timeout,
                    headers={"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.8"},
                    stream=True,
                )
                res["reachable"] = resp.status_code < 400
                if resp.status_code == 200 and "html" in resp.headers.get("Content-Type", "html"):
                    # Metadata lives in <head>; never download huge pages
                    html = resp.raw.read(600_000, decode_content=True).decode(resp.encoding or "utf-8", errors="ignore")
                res["last_modified"] = cls._parse_http_date(resp.headers.get("Last-Modified") or "")
                resp.close()
            except Exception:
                res["reachable"] = False

        if html:
            dt = cls._extract_date_from_html(html)
            if dt:
                res["published_at"] = dt
            res["caption"] = cls._extract_title_or_caption(html)
            res["headline"] = cls._extract_headline(html)

        return res

    @classmethod
    def classify_platform(cls, url: str) -> Tuple[str, float]:
        """
        Identifies platform and initial credibility rating based on domain.
        """
        platform, _, trust = cls.classify_url(url)
        return platform, trust

    @classmethod
    def classify_url(cls, url: str) -> Tuple[str, str, float]:
        """Returns (platform name, category, trust score) for a URL."""
        try:
            domain = cls.domain_of(url)
            if not domain:
                return "Unknown Web Platform", "web", 0.50

            for known_domain, (plat_name, category, trust) in cls.PLATFORM_DOMAINS.items():
                if known_domain in domain:
                    return plat_name, category, trust

            for news_word in cls.NEWS_INDICATORS:
                if news_word in domain:
                    return f"News: {domain}", "news", 0.85

            if "photography" in domain or "portfolio" in domain or "gallery" in domain:
                return f"Photographer: {domain}", "photo", 0.90
            if "fact" in domain or "verif" in domain:
                return f"Fact-check: {domain}", "fact_check", 0.90
            if domain.endswith(".gov") or domain.endswith(".edu") or ".gov." in domain or ".ac." in domain:
                return f"Institution: {domain}", "reference", 0.90
            if "wallpaper" in domain or "meme" in domain:
                return f"Wallpaper/meme site: {domain}", "blog", 0.25

            return f"Website: {domain}", "web", 0.50
        except Exception:
            return "Unknown Web Platform", "web", 0.50

    @staticmethod
    def domain_of(url: str) -> str:
        try:
            domain = urllib.parse.urlparse(url).netloc.lower()
            return domain[4:] if domain.startswith("www.") else domain
        except Exception:
            return ""

    @classmethod
    def _extract_date_from_html(cls, html: str) -> Optional[datetime]:
        for pattern in cls.DATE_PATTERNS:
            match = pattern.search(html)
            if match:
                raw_date = match.group(1).strip()
                dt = cls._parse_iso_date(raw_date)
                if dt:
                    return dt
        return None

    @classmethod
    def _extract_title_or_caption(cls, html: str) -> Optional[str]:
        for prop in ("og:description", "twitter:description", "description"):
            m = re.search(
                r'<meta[^>]+(?:property|name)=["\']' + re.escape(prop) + r'["\'][^>]+content=["\']([^"\']+)["\']',
                html, re.IGNORECASE,
            )
            if m:
                return html_lib.unescape(m.group(1).strip())[:400]
        return cls._extract_headline(html)

    @staticmethod
    def _extract_headline(html: str) -> Optional[str]:
        m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if not m:
            m = re.search(r"<title[^>]*>([^<]{3,300})</title>", html, re.IGNORECASE)
        return html_lib.unescape(m.group(1).strip())[:300] if m else None

    @staticmethod
    def _parse_iso_date(date_str: str) -> Optional[datetime]:
        clean = date_str.strip().replace("Z", "+00:00")
        dt = None
        try:
            dt = datetime.fromisoformat(clean)
        except Exception:
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
                try:
                    dt = datetime.strptime(date_str.strip()[:19], fmt)
                    break
                except Exception:
                    continue
        if dt is None:
            return None
        # Store naive UTC so timestamps compare cleanly with the rest of the engine
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        if dt.year < 1990 or dt > datetime.utcnow():
            return None
        return dt

    @staticmethod
    def _parse_http_date(date_str: str) -> Optional[datetime]:
        try:
            dt = parsedate_to_datetime(date_str)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return None
