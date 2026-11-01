"""Web image search & downloader (Bing with DuckDuckGo fallback) with safety filters.

Why this design
---------------
* Bing's legacy ``/images/async`` endpoint now returns unrelated "trending" results
  for most multi-word queries when called without a browser session (e.g.
  ``"Michael Jordan" slam dunk`` returned shoebill storks). The regular
  ``/images/search`` results page is still reliable, so we parse that instead.
* Negative keywords (``-sex -adult ...``) and exact-phrase quotes appended to the
  query badly degrade relevance on Bing. Safety is enforced instead with
  ``adlt=strict``, a URL/title blocklist and the optional moderation validator.
* Candidates are ranked by how well their page title / URL mention the target
  concept, so images that are actually about "Michael Jordan" come first.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from html import unescape
from io import BytesIO
import json
import os
import re
from typing import Any, Callable
import urllib.parse
import urllib.request

from PIL import Image

from stimulus_forge.services.openai_images import image_name_stem


ImageValidator = Callable[[bytes, str], bool]
ImageSavedCallback = Callable[[str], None]
ContinueCallback = Callable[[], bool]

_STOPWORDS = {
    "the", "and", "for", "with", "from", "of", "a", "an", "in", "on", "at", "to", "by",
}


def _clean(value: str) -> str:
    return " ".join(str(value or "").replace('"', "").strip().split())


def format_bing_term(term: str) -> str:
    """Normalize a search term (kept for backwards compatibility; no quoting)."""
    return _clean(term)


def _extract_disambiguation_hint(context: str, concept: str) -> str:
    """Extract a concise category or semantic disambiguation hint from context/excerpt."""
    if not context or not isinstance(context, str):
        return ""

    clean_ctx = _clean(context)
    if not clean_ctx:
        return ""

    # Check for 'Category: excerpt' pattern
    category_part = clean_ctx.split(":", 1)[0] if ":" in clean_ctx else clean_ctx
    cat_lower = re.sub(r"\s*\([^)]*\)", "", category_part).strip().lower()
    cat_lower = re.sub(r"^mentioned in document[:\s]*", "", cat_lower).strip()

    c_lower = concept.strip().lower()

    if any(k in cat_lower for k in ["tv show", "tv series", "television", "sitcom", "show"]):
        hint = "tv show"
    elif any(k in cat_lower for k in ["movie", "film", "cinema"]):
        hint = "movie"
    elif any(k in cat_lower for k in ["band", "music", "singer", "album", "musician", "song"]):
        hint = "band"
    elif any(k in cat_lower for k in ["sport", "sports", "team", "nba", "nfl", "mlb", "nhl"]):
        hint = "sports"
    elif any(k in cat_lower for k in ["book", "novel", "author"]):
        hint = "book"
    elif any(k in cat_lower for k in ["video game", "videogame", "gaming"]):
        hint = "video game"
    elif any(k in cat_lower for k in ["food", "dish", "cuisine", "recipe"]):
        hint = "food"
    elif any(k in cat_lower for k in ["holiday", "celebration"]):
        hint = "holiday"
    elif any(k in cat_lower for k in ["actor", "actress", "celebrity"]):
        hint = "actor"
    else:
        words = cat_lower.split()
        if 1 <= len(words) <= 3 and cat_lower != c_lower:
            hint = cat_lower
        else:
            hint = ""

    if hint and hint in c_lower:
        return ""
    return hint


def build_bing_query(
    concept: str,
    related_concept: str = "",
    name: str = "",
    context: str = "",
) -> str:
    """Build a natural-language image query for a concept / association.

    Search engines rank natural queries far better than quoted/negated ones, so
    the query is simply ``<context> <association>`` with duplicates removed.
    """
    c_raw = _clean(concept)
    rc_raw = _clean(related_concept)
    n_raw = _clean(name)
    hint = _extract_disambiguation_hint(context, c_raw)

    # Deduplicate repeated context between related_concept and name
    if rc_raw and n_raw:
        if rc_raw.casefold() == n_raw.casefold():
            n_raw = ""
        elif rc_raw.casefold() in n_raw.casefold():
            rc_raw = ""
        elif n_raw.casefold() in rc_raw.casefold():
            n_raw = ""

    # Character + performer (e.g. Michael Scott + Steve Carell)
    if rc_raw and n_raw:
        return f"{rc_raw} {n_raw}"

    assoc = rc_raw or n_raw
    if not assoc:
        if hint and hint not in c_raw.casefold():
            return f"{c_raw} {hint}"
        return c_raw
    if not c_raw or c_raw.casefold() == assoc.casefold():
        if hint and hint not in assoc.casefold():
            return f"{assoc} {hint}"
        return assoc

    # Association already contains the full concept (e.g. "Michael Jordan rookie card")
    if c_raw.casefold() in assoc.casefold():
        return assoc

    words = assoc.split()
    # Well-known standalone entities: acronyms ("NBA") or 3+ word proper names
    if (assoc.isupper() and len(assoc) <= 6) or (
        len(words) >= 3 and all(w[:1].isupper() for w in words)
    ):
        return assoc

    return f"{c_raw} {assoc}"


def _key_tokens(*texts: str) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        for tok in re.findall(r"[a-z0-9]+", _clean(text).casefold()):
            if len(tok) > 2 and tok not in _STOPWORDS and tok not in tokens:
                tokens.append(tok)
    return tokens


class BingDownloader:
    """Download normalized PNG images for a query across Wikimedia, DuckDuckGo, Openverse, and Bing."""

    DEFAULT_SAFE_QUERY_EXCLUSIONS = "-porn -pornography -xxx -nude -nudity -sex -adult"
    UNSAFE_URL_PATTERN = re.compile(
        r"(?:^|[./?&=_-])(porn|pornography|xxx|nude|nudity|nsfw)(?:[./?&=_-]|$)",
        re.IGNORECASE,
    )
    NON_PHOTO_PATTERN = re.compile(
        r"(?:^|[^a-z0-9])(logo|wordmark|icon|map|diagram|chart|graph|signature|autograph|plaque|"
        r"coat[ _-]+of[ _-]+arms|flag[ _-]+of|seal[ _-]+of|stamp[ _-]+of|grave|tombstone|"
        r"headstone|document|certificate|patent|schematic|blueprint|waveform|spectrogram)(?:[^a-z0-9]|$)",
        re.IGNORECASE,
    )
    WATERMARK_DOMAINS = (
        "alamy.com",
        "dreamstime.com",
        "shutterstock.com",
        "depositphotos.com",
        "123rf.com",
        "istockphoto.com",
        "gettyimages.com",
        "bigstockphoto.com",
        "vectorstock.com",
    )
    WIKIMEDIA_USER_AGENT = (
        "StimStudio/1.0 (https://github.com/stimstudio; contact@stimstudio.org)"
    )
    MAX_IMAGE_BYTES = 20 * 1024 * 1024
    MAX_IMAGE_PIXELS = 25_000_000
    MIN_IMAGE_EDGE = 160
    MAX_PAGES = 3
    FETCH_WORKERS = 6

    def __init__(
        self,
        query: str,
        concept: str,
        related_concept: str,
        name: str,
        limit: int,
        output_dir: str,
        timeout: int,
        filter: str = "",
        verbose: bool = True,
        safe_search: str = "strict",
        safe_query_exclusions: str | None = None,
        image_validator: ImageValidator | None = None,
        on_image: ImageSavedCallback | None = None,
        should_continue: ContinueCallback | None = None,
        seen_images: set[str] | None = None,
        seen_images_lock: Any | None = None,
        provider: str = "bing",
        min_width: int = 160,
        min_height: int = 160,
        context: str = "",
    ):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if not isinstance(timeout, int) or timeout < 1:
            raise ValueError("timeout must be a positive integer")
        if (safe_search or "").casefold() != "strict":
            raise ValueError("safe_search must be strict")

        exclusions = (
            safe_query_exclusions
            if safe_query_exclusions is not None
            else self.DEFAULT_SAFE_QUERY_EXCLUSIONS
        )
        # Exclusion words are applied as a title/URL blocklist, NOT appended to the
        # query (negative terms make Bing return unrelated results).
        self.blocked_terms = {
            w.lstrip("-").casefold() for w in (exclusions or "").split() if w.lstrip("-")
        }
        self.download_count = 0
        self.query = _clean(query)
        self.concept = concept
        self.related_concept = related_concept or ""
        self.name = name or ""
        self.limit = limit
        self.output_dir = output_dir
        self.safe_search = "strict"
        self.timeout = timeout
        self.filter = filter
        self.verbose = verbose
        self.image_validator = image_validator
        self.on_image = on_image
        self.should_continue = should_continue
        self.seen_urls: set[str] = set()
        self.seen_images = seen_images if seen_images is not None else set()
        self.seen_images_lock = seen_images_lock
        self.page_counter = 0
        self.offset = 0
        self.candidate_count = 0
        self.max_candidates = max(40, limit * 8)
        self.titles: dict[str, str] = {}
        self.thumbnails: dict[str, str] = {}
        self.provider = (provider or "bing").lower()
        self.source = self.provider
        self.min_width = max(16, int(min_width if min_width else self.MIN_IMAGE_EDGE))
        self.min_height = max(16, int(min_height if min_height else self.MIN_IMAGE_EDGE))
        # Tokens used to rank candidates: specific association/entity tokens weigh most.
        self.concept_tokens = _key_tokens(concept)
        self.assoc_tokens = [
            t for t in _key_tokens(related_concept, name) if t not in self.concept_tokens
        ]
        raw_assoc_tokens = _key_tokens(related_concept, name)
        self.context = context or ""
        hint = _extract_disambiguation_hint(self.context, concept)
        hint_tokens = _key_tokens(hint) if hint else []
        self.target_tokens = raw_assoc_tokens if raw_assoc_tokens else list(self.concept_tokens)
        self.context_tokens = [
            t for t in self.concept_tokens if t not in self.target_tokens
        ] + [t for t in hint_tokens if t not in self.target_tokens]
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cookie": "SRCHHPGUSR=ADLT=STRICT",
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def get_filter(shorthand: str | None) -> str:
        filters = {
            "line": "+filterui:photo-linedrawing",
            "linedrawing": "+filterui:photo-linedrawing",
            "photo": "+filterui:photo-photo",
            "clipart": "+filterui:photo-clipart",
            "gif": "+filterui:photo-animatedgif",
            "animatedgif": "+filterui:photo-animatedgif",
            "transparent": "+filterui:photo-transparent",
            "square": "+filterui:imagesize-custom_600_600+filterui:aspect-square",
            "square+photo": (
                "+filterui:imagesize-custom_600_600+filterui:aspect-square"
                "+filterui:photo-photo"
            ),
            "square+face": (
                "+filterui:imagesize-custom_600_600+filterui:aspect-square"
                "+filterui:face-face"
            ),
        }
        return filters.get(shorthand or "", "")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _is_unsafe(self, link: str, title: str = "") -> bool:
        unquoted = urllib.parse.unquote(link)
        if self.UNSAFE_URL_PATTERN.search(unquoted):
            return True
        words = set(re.findall(r"[a-z]+", f"{title} {unquoted}".casefold()))
        return bool(words & self.blocked_terms)

    def _is_non_photo(self, link: str, title: str = "") -> bool:
        if self.filter in {"line", "linedrawing", "clipart", "transparent", "gif", "animatedgif"}:
            return False
        unquoted_path = urllib.parse.urlsplit(urllib.parse.unquote(link)).path.casefold()
        if unquoted_path.endswith((".svg", ".tif", ".tiff", ".pdf", ".djvu", ".gif", ".ico")):
            return True
        if ".svg/" in unquoted_path or unquoted_path.endswith(".svg.png"):
            return True
        combined = f"{title} {os.path.basename(unquoted_path)}"
        return bool(self.NON_PHOTO_PATTERN.search(combined))

    def _token_present(self, tok: str, words: set[str], stem_compact: str) -> bool:
        if tok in words:
            return True
        # Allow concatenated filename stems like "michaelscott.png" -> "michaelscott"
        return len(tok) >= 4 and tok in stem_compact

    def _matched_tokens(self, link: str, tokens: list[str]) -> int:
        unquoted = urllib.parse.unquote(link)
        raw_text = f"{self.titles.get(link, '')} {unquoted}".casefold()
        words = set(re.findall(r"[a-z0-9]+", raw_text))
        filename_stem = os.path.splitext(os.path.basename(urllib.parse.urlsplit(unquoted).path))[0].casefold()
        stem_compact = filename_stem if " " not in filename_stem and "_" not in filename_stem and "-" not in filename_stem else ""
        return sum(1 for tok in tokens if self._token_present(tok, words, stem_compact))

    def _is_relevant(self, link: str) -> bool:
        """Return True if the candidate genuinely matches the target concept/association."""
        if not self.target_tokens:
            return True
        target_hits = self._matched_tokens(link, self.target_tokens)
        context_hits = self._matched_tokens(link, self.context_tokens)
        if len(self.target_tokens) == 1:
            return target_hits >= 1
        # When 2+ target tokens exist (e.g. "Michael Jordan" or "Michael Scott Steve Carell"),
        # require at least 2 target token matches (or 1 target + 1 parent context token) so
        # single-name false positives like "Michael Jackson" are rejected.
        return target_hits >= 2 or (target_hits >= 1 and context_hits >= 1)

    def _relevance(self, link: str) -> int:
        title = self.titles.get(link, "")
        unquoted = urllib.parse.unquote(link)
        raw_text = f"{title} {unquoted}".casefold()
        words = set(re.findall(r"[a-z0-9]+", raw_text))
        filename_stem = os.path.splitext(os.path.basename(urllib.parse.urlsplit(unquoted).path))[0].casefold()
        stem_compact = filename_stem if " " not in filename_stem and "_" not in filename_stem and "-" not in filename_stem else ""

        def _has(tok: str) -> bool:
            return self._token_present(tok, words, stem_compact)

        score = 0
        target_hits = 0
        for tok in self.target_tokens:
            if _has(tok):
                score += 4
                target_hits += 1
        for tok in self.context_tokens:
            if tok in words:
                score += 3

        # Bonus when all tokens of a multi-word entity match together
        if len(self.target_tokens) >= 2 and target_hits >= 2:
            score += 5
        if self.concept_tokens and all(t in words for t in self.concept_tokens):
            score += 3

        # Penalize heavily watermarked stock agencies and non-photographic assets
        host = urllib.parse.urlsplit(link).netloc.casefold()
        if any(domain in host for domain in self.WATERMARK_DOMAINS):
            score -= 4
        if self._is_non_photo(link, title):
            score -= 8
        return score

    @staticmethod
    def _perceptual_hash(image_bytes: bytes) -> str | None:
        """Compute a 64-bit difference hash (dHash) to detect resized/recompressed duplicates."""
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                resample = getattr(Image, "Resampling", Image).LANCZOS
                gray = img.convert("L").resize((9, 8), resample)
                pixels = list(getattr(gray, "get_flattened_data", gray.getdata)())
                bits = 0
                for row in range(8):
                    offset = row * 9
                    for col in range(8):
                        bits = (bits << 1) | (1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
                return f"dhash:{bits:016x}"
        except Exception:
            return None

    def _request_headers_for_url(self, link: str) -> dict[str, str]:
        host = urllib.parse.urlsplit(link).netloc.casefold()
        if host.endswith(("wikimedia.org", "wikipedia.org")):
            return {
                "User-Agent": self.WIKIMEDIA_USER_AGENT,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            }
        return self.headers

    def _fetch_and_normalize(self, link: str) -> bytes:
        if urllib.parse.urlsplit(link).scheme not in {"http", "https"}:
            raise ValueError("unsupported image URL scheme")
        request = urllib.request.Request(link, None, self._request_headers_for_url(link))
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > self.MAX_IMAGE_BYTES:
                raise ValueError("image exceeds the 20 MB safety limit")
            raw = response.read(self.MAX_IMAGE_BYTES + 1)
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
        if len(raw) > self.MAX_IMAGE_BYTES:
            raise ValueError("image exceeds the 20 MB safety limit")
        if content_type and not content_type.startswith("image/"):
            raise ValueError(f"URL returned {content_type}, not an image")

        with Image.open(BytesIO(raw)) as source:
            if source.width * source.height > self.MAX_IMAGE_PIXELS:
                raise ValueError("image dimensions exceed the safety limit")
            source.load()
            if source.width < self.min_width or source.height < self.min_height:
                raise ValueError(
                    f"image ({source.width}x{source.height}) is smaller than {self.min_width}x{self.min_height}px"
                )
            if max(source.size) / min(source.size) > 4:
                raise ValueError("image aspect ratio is too extreme")
            normalized = source.convert("RGBA" if "A" in source.getbands() else "RGB")
            output = BytesIO()
            normalized.save(output, format="PNG", optimize=True)
        return output.getvalue()

    def _try_fetch(self, link: str) -> bytes | None:
        try:
            return self._fetch_and_normalize(link)
        except Exception as exc:
            fallback_url = self.thumbnails.get(link)
            if fallback_url and fallback_url != link:
                try:
                    return self._fetch_and_normalize(fallback_url)
                except Exception:
                    pass
            self._log(f"[!] Skipped {link}: {exc}")
            return None

    def _save_bytes(self, link: str, image_bytes: bytes) -> bool:
        fingerprint = hashlib.sha256(image_bytes).hexdigest()
        dhash = self._perceptual_hash(image_bytes)
        if self.image_validator is not None and not self.image_validator(
            image_bytes, "image/png"
        ):
            self._log(f"[Safety] Rejected image from {link}")
            return False

        if self.seen_images_lock is not None:
            with self.seen_images_lock:
                if fingerprint in self.seen_images or (dhash and dhash in self.seen_images):
                    return False
                self.seen_images.add(fingerprint)
                if dhash:
                    self.seen_images.add(dhash)
        else:
            if fingerprint in self.seen_images or (dhash and dhash in self.seen_images):
                return False
            self.seen_images.add(fingerprint)
            if dhash:
                self.seen_images.add(dhash)

        self.download_count += 1
        stem = image_name_stem(self.concept, self.related_concept, self.name)
        image_index = 1
        while os.path.exists(os.path.join(self.output_dir, f"{stem}_{image_index}.png")):
            image_index += 1
        file_path = os.path.join(self.output_dir, f"{stem}_{image_index}.png")
        with open(file_path, "wb") as image_file:
            image_file.write(image_bytes)
        if self.on_image is not None:
            self.on_image(file_path)
        self._log(f"Saved Image #{self.download_count} to {file_path}")
        return True

    def download_image(self, link: str) -> bool:
        """Validate, optionally moderate, and save one candidate URL."""
        try:
            if self._is_unsafe(link, self.titles.get(link, "")):
                self._log(f"[Safety] Rejected unsafe image URL {link}")
                return False
            self._log(f"Checking candidate image from {link}")
            image_bytes = self._fetch_and_normalize(link)
            return self._save_bytes(link, image_bytes)
        except Exception as exc:
            fallback_url = self.thumbnails.get(link)
            if fallback_url and fallback_url != link:
                try:
                    image_bytes = self._fetch_and_normalize(fallback_url)
                    return self._save_bytes(link, image_bytes)
                except Exception:
                    pass
            self._log(f"[!] Skipped {link}: {exc}")
            return False

    # ------------------------------------------------------------------ search
    def _parse_results(self, html: str) -> list[str]:
        links: list[str] = []
        # Modern results page: <a class="iusc" m="{json with murl/t/turl}">
        for meta in re.findall(r'class="iusc"[^>]*?\bm="([^"]+)"', html):
            try:
                data = json.loads(unescape(meta))
            except ValueError:
                continue
            murl = data.get("murl")
            if murl:
                links.append(murl)
                self.titles.setdefault(murl, data.get("t", "") or "")
                turl = data.get("turl")
                if turl:
                    self.thumbnails.setdefault(murl, turl)
        if not links:
            # Legacy markup fallback
            links = [unescape(link) for link in re.findall(r"murl&quot;:&quot;(.*?)&quot;", html)]
        return links

    def _search_page(self) -> list[str]:
        """Fetch one page of Bing image results (strict SafeSearch)."""
        parameters = urllib.parse.urlencode(
            {
                "q": self.query,
                "first": self.offset + 1,
                "form": "HDRSC2",
                "adlt": self.safe_search,
                "qft": self.get_filter(self.filter),
            }
        )
        request = urllib.request.Request(
            f"https://www.bing.com/images/search?{parameters}", None, headers=self.headers
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
        links = self._parse_results(html)
        self.offset += max(len(links), 1)
        return links

    def _search_duckduckgo(self) -> list[str]:
        """Search DuckDuckGo images (strict safe search)."""
        headers = dict(self.headers)
        headers["Cookie"] = "kp=1"
        page_url = "https://duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": self.query, "iax": "images", "ia": "images", "kp": "1"}
        )
        with urllib.request.urlopen(
            urllib.request.Request(page_url, None, headers), timeout=self.timeout
        ) as response:
            html = response.read().decode("utf-8", errors="replace")
        token = re.search(r"vqd=['\"]?([\d-]+)", html)
        if not token:
            return []
        headers.update({"Referer": "https://duckduckgo.com/", "Accept": "application/json"})
        api_url = "https://duckduckgo.com/i.js?" + urllib.parse.urlencode(
            {"q": self.query, "o": "json", "vqd": token.group(1), "p": "1", "l": "us-en"}
        )
        with urllib.request.urlopen(
            urllib.request.Request(api_url, None, headers), timeout=self.timeout
        ) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
        links = []
        for item in data.get("results", []):
            url = item.get("image")
            thumb = item.get("thumbnail")
            item_w = int(item.get("width") or 0)
            item_h = int(item.get("height") or 0)
            if item_w > 0 and item_h > 0 and (item_w < self.min_width or item_h < self.min_height):
                continue
            if url:
                links.append(url)
                self.titles.setdefault(url, item.get("title", "") or "")
                if thumb:
                    self.thumbnails.setdefault(url, thumb)
        return links

    def _entity_search_queries(self) -> list[str]:
        """Ordered list of entity queries from most specific to broader fallback."""
        queries: list[str] = []
        rc = _clean(self.related_concept)
        nm = _clean(self.name)
        c = _clean(self.concept)

        def _add(q: str) -> None:
            q_clean = _clean(q)
            if q_clean and all(q_clean.casefold() != x.casefold() for x in queries):
                queries.append(q_clean)

        if rc and nm:
            # Fictional role + performer (either order from UI/API)
            _add(f"{rc} {c}" if c else rc)
            _add(nm)
            _add(rc)
        elif rc or nm:
            assoc = rc or nm
            _add(self.query)
            _add(assoc)
        else:
            _add(self.query)
            _add(c)
        return queries

    def _search_wikipedia(self) -> list[str]:
        """Fetch curated Wikipedia article lead/infobox images for the target entity."""
        headers = {
            "User-Agent": self.WIKIMEDIA_USER_AGENT,
            "Accept": "application/json",
        }
        links: list[str] = []
        thumb_size = max(1024, self.min_width, self.min_height)
        for q in self._entity_search_queries():
            if len(links) >= self.limit * 2:
                break
            parameters = urllib.parse.urlencode(
                {
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": q,
                    "gsrlimit": 5,
                    "prop": "pageimages",
                    "piprop": "original|thumbnail",
                    "pithumbsize": thumb_size,
                    "pilicense": "any",
                    "format": "json",
                }
            )
            api_url = f"https://en.wikipedia.org/w/api.php?{parameters}"
            try:
                request = urllib.request.Request(api_url, None, headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8", errors="replace"))
                pages = sorted(
                    data.get("query", {}).get("pages", {}).values(),
                    key=lambda p: p.get("index", 9999),
                )
                for page in pages:
                    title = page.get("title", "")
                    orig_info = page.get("original") or {}
                    thumb_info = page.get("thumbnail") or {}
                    orig_url = orig_info.get("source")
                    thumb_url = thumb_info.get("source")
                    orig_w = int(orig_info.get("width") or 0)
                    orig_h = int(orig_info.get("height") or 0)
                    if orig_w > 0 and orig_h > 0 and (orig_w < self.min_width or orig_h < self.min_height):
                        continue
                    img_url = thumb_url or orig_url
                    if not img_url or self._is_non_photo(img_url, title):
                        continue
                    if img_url not in links:
                        links.append(img_url)
                        self.titles.setdefault(img_url, title)
                        if orig_url and orig_url != img_url:
                            self.thumbnails.setdefault(img_url, orig_url)
            except Exception as exc:
                self._log(f"[!] Wikipedia pageimages search failed for {q!r}: {exc}")
        return links

    def _commons_search_queries(self) -> list[str]:
        """Build targeted Wikimedia Commons queries prioritizing file title matches."""
        rc = _clean(self.related_concept)
        nm = _clean(self.name)
        c = _clean(self.concept)
        queries: list[str] = []

        def _add(q: str) -> None:
            q_clean = q.strip()
            if q_clean and q_clean not in queries:
                queries.append(q_clean)

        if rc and nm:
            _add(f'intitle:"{nm}" filetype:bitmap')
            _add(f'intitle:"{rc}" filetype:bitmap')
            _add(self.query)
        elif rc or nm:
            assoc = rc or nm
            if c and c.casefold() not in assoc.casefold():
                _add(f'intitle:"{assoc}" {c} filetype:bitmap')
                _add(f'intitle:"{c}" {assoc} filetype:bitmap')
            _add(f'intitle:"{assoc}" filetype:bitmap')
            _add(self.query)
        else:
            _add(f'intitle:"{self.query}" filetype:bitmap')
            _add(self.query)
        return queries

    def _search_wikimedia(self) -> list[str]:
        """Search Wikimedia Commons for authentic, high-quality, vetted images."""
        headers = {
            "User-Agent": self.WIKIMEDIA_USER_AGENT,
            "Accept": "application/json",
        }
        links: list[str] = []
        seen_base_titles: set[str] = set()
        thumb_width = max(1280, self.min_width, self.min_height)

        for q in self._commons_search_queries():
            if len(links) >= self.max_candidates:
                break
            parameters = urllib.parse.urlencode(
                {
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": q,
                    "gsrnamespace": 6,
                    "gsrlimit": min(30, self.max_candidates),
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime",
                    "iiurlwidth": thumb_width,
                    "format": "json",
                }
            )
            api_url = f"https://commons.wikimedia.org/w/api.php?{parameters}"
            try:
                request = urllib.request.Request(api_url, None, headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8", errors="replace"))
                pages = sorted(
                    data.get("query", {}).get("pages", {}).values(),
                    key=lambda p: p.get("index", 9999),
                )
                for page in pages:
                    title = page.get("title", "")
                    infos = page.get("imageinfo", [])
                    if not infos:
                        continue
                    info = infos[0]
                    mime = info.get("mime", "")
                    if mime not in {"image/jpeg", "image/png", "image/webp"}:
                        continue
                    orig_w = int(info.get("width") or 0)
                    orig_h = int(info.get("height") or 0)
                    if orig_w > 0 and orig_h > 0 and (orig_w < self.min_width or orig_h < self.min_height):
                        continue
                    clean_title = re.sub(r"^File:\s*", "", title, flags=re.IGNORECASE)
                    if self._is_non_photo(info.get("url", ""), clean_title):
                        continue
                    # Avoid saving multiple crops of the exact same Commons upload
                    base_title = re.sub(
                        r"\s*\([^)]*crop[^)]*\)", "", clean_title, flags=re.IGNORECASE
                    ).casefold()
                    # Prefer thumb if it meets min_width/min_height, otherwise original
                    thumb_url = info.get("thumburl")
                    orig_url = info.get("url")
                    thumb_w = int(info.get("thumbwidth") or 0)
                    thumb_h = int(info.get("thumbheight") or 0)
                    if thumb_url and thumb_w > 0 and thumb_h > 0 and (thumb_w < self.min_width or thumb_h < self.min_height):
                        img_url = orig_url or thumb_url
                    else:
                        img_url = thumb_url or orig_url
                    if img_url and img_url not in links and base_title not in seen_base_titles:
                        seen_base_titles.add(base_title)
                        links.append(img_url)
                        self.titles.setdefault(img_url, clean_title)
                        if orig_url and orig_url != img_url:
                            self.thumbnails.setdefault(img_url, orig_url)
            except Exception as exc:
                self._log(f"[!] Wikimedia search failed for {q!r}: {exc}")
        return links

    def _search_openverse(self) -> list[str]:
        """Search Openverse (Creative Commons, museum collections, Flickr)."""
        headers = {
            "User-Agent": self.WIKIMEDIA_USER_AGENT,
            "Accept": "application/json",
        }
        # Openverse anonymous API enforces page_size <= 20 (returns HTTP 401 if > 20)
        parameters = urllib.parse.urlencode(
            {
                "q": self.query,
                "page_size": min(20, self.max_candidates),
                "mature": "false",
            }
        )
        links: list[str] = []
        for base_host in ("https://api.openverse.org", "https://api.openverse.engineering"):
            api_url = f"{base_host}/v1/images/?{parameters}"
            try:
                request = urllib.request.Request(api_url, None, headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8", errors="replace"))
                for item in data.get("results", []):
                    url = item.get("url")
                    thumb = item.get("thumbnail")
                    title = item.get("title", "") or ""
                    if url and url not in links and not self._is_non_photo(url, title):
                        links.append(url)
                        self.titles.setdefault(url, title)
                        if thumb:
                            self.thumbnails.setdefault(url, thumb)
                if links:
                    break
            except Exception as exc:
                self._log(f"[!] Openverse search failed on {base_host} for {self.query!r}: {exc}")
        return links

    def _add_unique(self, target: list[str], links: list[str]) -> None:
        for link in links:
            if link and link not in self.seen_urls:
                self.seen_urls.add(link)
                target.append(link)

    def _collect_candidates(self) -> list[str]:
        candidates: list[str] = []
        min_relevant_goal = max(self.limit * 2, 4)

        def _relevant_count() -> int:
            return sum(
                1
                for c in candidates
                if not self._is_unsafe(c, self.titles.get(c, "")) and self._is_relevant(c)
            )

        if self.provider == "wikimedia":
            self._log(f"Searching Wikipedia & Wikimedia Commons for {self.query!r}")
            self._add_unique(candidates, self._search_wikipedia())
            self._add_unique(candidates, self._search_wikimedia())
            if _relevant_count() < min_relevant_goal:
                self._log("Wikimedia returned few relevant results; checking Openverse & DuckDuckGo")
                self._add_unique(candidates, self._search_openverse())
            if _relevant_count() < min_relevant_goal:
                try:
                    self._add_unique(candidates, self._search_duckduckgo())
                except Exception as exc:
                    self._log(f"[!] DuckDuckGo fallback failed: {exc}")

        elif self.provider == "openverse":
            self._log(f"Searching Openverse for {self.query!r}")
            self._add_unique(candidates, self._search_openverse())
            if _relevant_count() < min_relevant_goal:
                self._log("Openverse returned few relevant results; checking Wikipedia & Wikimedia")
                self._add_unique(candidates, self._search_wikipedia())
                self._add_unique(candidates, self._search_wikimedia())
            if _relevant_count() < min_relevant_goal:
                try:
                    self._add_unique(candidates, self._search_duckduckgo())
                except Exception as exc:
                    self._log(f"[!] DuckDuckGo fallback failed: {exc}")

        elif self.provider == "duckduckgo":
            self._log(f"Searching DuckDuckGo & Wikipedia for {self.query!r}")
            try:
                self._add_unique(candidates, self._search_duckduckgo())
            except Exception as exc:
                self._log(f"[!] DuckDuckGo search failed: {exc}")
            self._add_unique(candidates, self._search_wikipedia())
            if _relevant_count() < min_relevant_goal:
                self._add_unique(candidates, self._search_wikimedia())

        else:  # bing
            while self.page_counter < self.MAX_PAGES and len(candidates) < self.max_candidates:
                self._log(f"Indexing Bing image page {self.page_counter + 1} for {self.query!r}")
                try:
                    links = self._search_page()
                except Exception as exc:
                    self._log(f"[!] Bing search failed: {exc}")
                    links = []
                self.page_counter += 1
                before = len(candidates)
                self._add_unique(candidates, links)
                if len(candidates) == before or _relevant_count() >= min_relevant_goal:
                    break

            if _relevant_count() < min_relevant_goal:
                self._log("Bing returned few relevant results; checking DuckDuckGo & Wikipedia")
                try:
                    self._add_unique(candidates, self._search_duckduckgo())
                except Exception as exc:
                    self._log(f"[!] DuckDuckGo search failed: {exc}")
                self._add_unique(candidates, self._search_wikipedia())
                if _relevant_count() < min_relevant_goal:
                    self._add_unique(candidates, self._search_wikimedia())

        safe_candidates = [
            c for c in candidates if not self._is_unsafe(c, self.titles.get(c, ""))
        ]
        relevant = [c for c in safe_candidates if self._is_relevant(c)]
        relevant.sort(key=self._relevance, reverse=True)

        if len(relevant) >= self.limit:
            return relevant[: self.max_candidates]

        # If strict matching yielded fewer than limit, append best-scoring remaining candidates
        remainder = [c for c in safe_candidates if c not in relevant and self._relevance(c) > 0]
        remainder.sort(key=self._relevance, reverse=True)
        return (relevant + remainder)[: self.max_candidates]

    def run(self) -> list[str]:
        os.makedirs(self.output_dir, exist_ok=True)
        saved: list[str] = []
        original_on_image = self.on_image

        def _record(path: str) -> None:
            saved.append(path)
            if original_on_image is not None:
                original_on_image(path)

        self.on_image = _record
        try:
            candidates = self._collect_candidates()
            batch_size = max(self.FETCH_WORKERS, self.limit * 2)
            with ThreadPoolExecutor(max_workers=self.FETCH_WORKERS) as pool:
                for start in range(0, len(candidates), batch_size):
                    if self.download_count >= self.limit:
                        break
                    if self.should_continue is not None and not self.should_continue():
                        break
                    batch = candidates[start:start + batch_size]
                    self.candidate_count += len(batch)
                    for link, image_bytes in zip(batch, pool.map(self._try_fetch, batch)):
                        if self.download_count >= self.limit:
                            break
                        if image_bytes is not None:
                            self._save_bytes(link, image_bytes)
        finally:
            self.on_image = original_on_image

        self._log(f"Done. Downloaded {self.download_count} validated images.")
        return saved

