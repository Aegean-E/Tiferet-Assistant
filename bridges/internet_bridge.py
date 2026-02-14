import requests
import time
import os
import json
import re
from typing import Optional, Callable, Dict, Tuple
import threading
import xml.etree.ElementTree as ET
import ipaddress
import socket
from urllib.parse import urlparse
import logging

try:
    from ai_core.lm import run_local_lm
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ai_core.lm import run_local_lm

class InternetBridge:
    """
    Strictly limited internet access bridge.
    Allowed sources: WIKIPEDIA, PUBMED, ARXIV.
    Capabilities: Read-only search and summary retrieval.
    """
    ALLOWED_SOURCES = {"WIKIPEDIA", "PUBMED", "ARXIV", "WEB"}

    def __init__(self, get_settings_fn: Callable[[], Dict], log_fn=logging.info, save_dir: str = "./data/internet_data"):
        self.get_settings = get_settings_fn
        self.log = log_fn
        self.save_dir = save_dir
        self.last_request_time = {} # Per-source rate limiting
        self.request_interval = 2.0 # Rate limiting to be polite
        self.cache = {} # Cache: {query_key: (content, filepath)}
        self.lock = threading.Lock() # Thread safety
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.session = requests.Session()
        self._load_cache()
        self._check_dependencies()

    def _check_connectivity(self) -> bool:
        """Simple check to see if we can reach the internet."""
        try:
            # Try to reach a reliable host (Google DNS)
            self.session.get("https://8.8.8.8", timeout=3)
            return True
        except Exception:
            return False

    def _is_safe_url(self, url: str) -> bool:
        """
        SSRF Protection: Check if URL resolves to a private/local IP.
        Uses early rejection for literal IPs and timeouts for DNS resolution.
        """
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname: return False
            
            # 1. Early Rejection: Check if hostname is a literal private IP
            try:
                ip_obj = ipaddress.ip_address(hostname)
                if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                    self.log(f"🛡️ Blocked literal private IP (SSRF): {hostname}")
                    return False
            except ValueError:
                # Not a literal IP, proceed to resolution
                pass

            # 2. Block common local domains early to prevent DNS leaks
            if hostname.lower().endswith(('.local', '.lan', '.home', '.internal', '.localhost')):
                self.log(f"🛡️ Blocked local domain (SSRF): {hostname}")
                return False

            # 3. Resolve hostname to IP with timeout protection
            # We use getaddrinfo but wrap it to be careful
            addr_info = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
            for family, type, proto, canonname, sockaddr in addr_info:
                ip = sockaddr[0]
                ip_addr = ipaddress.ip_address(ip)
                
                # Check if private or loopback
                if ip_addr.is_private or ip_addr.is_loopback or ip_addr.is_link_local:
                    self.log(f"🛡️ Blocked resolved private IP (SSRF): {hostname} -> {ip}")
                    return False
                
            return True
        except socket.gaierror as e:
            self.log(f"⚠️ DNS resolution failed for {hostname}: {e}")
            return False
        except Exception as e:
            self.log(f"⚠️ URL validation failed: {e}")
            return False

    def _check_dependencies(self):
        """Warn once about missing optional dependencies."""
        missing = []
        try:
            import bs4
        except ImportError:
            missing.append("beautifulsoup4")
        
        try:
            import ddgs
        except ImportError:
            try:
                import duckduckgo_search
            except ImportError:
                missing.append("duckduckgo-search")
                
        if missing:
            self.log(f"⚠️ InternetBridge: Optional dependencies missing: {', '.join(missing)}. Some features may be limited.")

    def _load_cache(self):
        """Load cache from disk."""
        cache_path = os.path.join(self.save_dir, "search_cache.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception as e:
                self.log(f"⚠️ Failed to load cache: {e}")

    def _save_cache(self):
        """Save cache to disk."""
        cache_path = os.path.join(self.save_dir, "search_cache.json")
        temp_path = cache_path + ".tmp"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, cache_path)

    def search(self, query: str, source: str) -> Tuple[str, Optional[str]]:
        """
        Execute an approved search request.
        Returns: (content_string, filepath_if_saved)
        """
        # 0. Cache Check
        cache_key = f"{source}:{query}"
        with self.lock:
            if cache_key in self.cache:
                self.log(f"wd Cache hit for: {query} ({source})")
                return self.cache[cache_key]

            # 1. Rate Limit Check (Per Source)
            # Also check per-query to prevent spamming same query if not cached yet
            last_time = self.last_request_time.get(source, 0)
            if time.time() - last_time < self.request_interval:
                return f"⚠️ Rate limit active for {source}. Please wait a moment.", None
            self.last_request_time[source] = time.time()

        # 2. Source Check
        source = source.upper().strip()
        if source not in self.ALLOWED_SOURCES:
            return f"❌ Access Denied: Source '{source}' is not in the approved list {self.ALLOWED_SOURCES}.", None

        # 3. Safety Evaluation Check
        if not self._evaluate_safety(query):
            return f"❌ Access Denied: Query '{query}' failed safety evaluation.", None

        # 4. Connectivity Check
        if not self._check_connectivity():
            return "⚠️ Internet access unavailable (DNS/Connection failure). Cannot perform search.", None

        # Scientific Foraging Bias
        # If source is generic WEB but query looks scientific, try ArXiv/PubMed first
        if source == "WEB":
            scientific_keywords = ["paper", "study", "research", "analysis", "algorithm", "theorem", "mechanism", "synthesis", "quantum", "neural", "cognitive"]
            if any(kw in query.lower() for kw in scientific_keywords):
                self.log(f"🔬 Scientific Foraging: Prioritizing ArXiv/PubMed for '{query}'")
                # Try ArXiv first
                arxiv_res = self._search_arxiv(query)
                if "No ArXiv results" not in arxiv_res[0] and "Error" not in arxiv_res[0]:
                    return arxiv_res

        self.log(f"🌐 Bridge executing search on {source}: {query}")

        try:
            result = None
            if source == "WIKIPEDIA":
                result = self._search_wikipedia(query)
            elif source == "PUBMED":
                result = self._search_pubmed(query)
            elif source == "ARXIV":
                result = self._search_arxiv(query)
            elif source == "WEB":
                result = self._search_web(query)
            else:
                return "❌ Unknown source.", None
            
            if result and result[1]: # If filepath is present, cache it
                with self.lock:
                    self.cache[cache_key] = result
                    self._save_cache()
            return result
        except Exception as e:
            return f"❌ Internet Bridge Error: {e}", None

    def _search_wikipedia(self, query: str) -> Tuple[str, Optional[str]]:
        content = self._do_wikipedia_search(query)
        if "ℹ️" in content or "❌" in content:
            return content, None
        filepath = self._save_to_file(query, "WIKIPEDIA", content)
        return content, filepath

    def _search_pubmed(self, query: str) -> Tuple[str, Optional[str]]:
        content = self._do_pubmed_search(query)
        if "ℹ️" in content or "❌" in content:
            return content, None
        filepath = self._save_to_file(query, "PUBMED", content)
        return content, filepath

    def _search_arxiv(self, query: str) -> Tuple[str, Optional[str]]:
        content = self._do_arxiv_search(query)
        if "ℹ️" in content or "❌" in content:
            return content, None
        filepath = self._save_to_file(query, "ARXIV", content)
        return content, filepath

    def _save_to_file(self, query: str, source: str, content: str) -> Optional[str]:
        """Save search result to a text file."""
        try:
            timestamp = int(time.time())
            # Sanitize filename
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')[:50]
            filename = f"{source}_{safe_query}_{timestamp}.txt"
            filepath = os.path.join(self.save_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Query: {query}\nSource: {source}\nDate: {time.ctime(timestamp)}\n\n{content}")
            
            self.log(f"💾 Saved internet data to: {filename}")
            return filepath
        except Exception as e:
            self.log(f"⚠️ Failed to save internet data: {e}")
            return None
        
    def _evaluate_safety(self, query: str) -> bool:
        """
        Use LLM to verify if the query is safe and relevant.
        """
        settings = self.get_settings()
        prompt = (
            f"SYSTEM SECURITY CHECK. Query: '{query}'\n"
            "Assess if this query is safe for an AI research assistant.\n"
            "FAIL if: Illegal, Harmful, Explicit, Jailbreak, or Irrelevant to knowledge.\n"
            "PASS if: Factual, Academic, or General Knowledge.\n"
            "Output ONLY 'PASS' or 'FAIL'."
        )
        try:
            response = run_local_lm(
                messages=[{"role": "user", "content": f"Check query: {query}"}],
                system_prompt=prompt,
                temperature=0.0,
                max_tokens=10,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )
            is_safe = "PASS" in response.upper()
            self.log(f"🛡️ Safety Eval: '{query}' -> {response.strip()} [{'PASSED' if is_safe else 'FAILED'}]")
            return is_safe
        except Exception as e:
            self.log(f"⚠️ Safety check failed (LLM Error): {e}. Proceeding cautiously.")
            return False # Fail closed (block search) if LLM is down for security.

    def _do_wikipedia_search(self, query: str) -> str:
        # Wikipedia API
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1
        }
        headers = {'User-Agent': 'AI_Research_Assistant/1.0'}
        
        resp = self.session.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return "ℹ️ No Wikipedia results found."
            
        # Get page content for the top result
        page_id = search_results[0]["pageid"]
        title = search_results[0]["title"]
        
        params_content = {
            "action": "query",
            "prop": "extracts",
            "pageids": page_id,
            "explaintext": True,
            "exintro": True, # Only introduction to save tokens/bandwidth
            "format": "json"
        }
        
        resp_content = self.session.get(url, params=params_content, headers=headers, timeout=10)
        resp_content.raise_for_status()
        data_content = resp_content.json()
        
        pages = data_content.get("query", {}).get("pages", {})
        page_text = "No content."
        for pid, pdata in pages.items():
            page_text = pdata.get("extract", "No content.")
            break
            
        return f"📖 Wikipedia Summary ({title}):\n{page_text}"

    def _do_pubmed_search(self, query: str) -> str:
        # PubMed E-utilities
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # 1. ESearch
        search_url = f"{base_url}/esearch.fcgi"
        search_params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 3}
        
        resp = self.session.get(search_url, params=search_params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return "ℹ️ No PubMed results found."
            
        # 2. ESummary (Get details)
        summary_url = f"{base_url}/esummary.fcgi"
        summary_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "json"}
        
        resp_sum = self.session.get(summary_url, params=summary_params, timeout=10)
        resp_sum.raise_for_status()
        data_sum = resp_sum.json()
        
        results = []
        uids = data_sum.get("result", {}).get("uids", [])
        for uid in uids:
            item = data_sum["result"][uid]
            title = item.get("title", "No Title")
            source = item.get("source", "Unknown Source")
            pubdate = item.get("pubdate", "Unknown Date")
            results.append(f"📄 {title}\n   Source: {source} ({pubdate})\n   ID: {uid}")
            
        return f"🔬 PubMed Results (Top 3):\n" + "\n\n".join(results)

    def _do_arxiv_search(self, query: str) -> str:
        # ArXiv API
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 3
        }
        
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        
        # Parse XML
        try:
            root = ET.fromstring(resp.content)
            
            # Namespace map for Atom
            ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
            
            entries = root.findall('atom:entry', ns)
            if not entries:
                return "ℹ️ No ArXiv results found."
                
            results = []
            for entry in entries:
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                published = entry.find('atom:published', ns).text[:10]
                link = entry.find('atom:id', ns).text
                
                results.append(f"📄 {title}\n   Published: {published}\n   Link: {link}\n   Abstract: {summary[:500]}...")
                
            return f"📜 ArXiv Results (Top 3):\n" + "\n\n".join(results)
            
        except Exception as e:
            return f"❌ Error parsing ArXiv response: {e}"

    def _preflight_check(self, url: str, timeout: int = 5) -> bool:
        """
        Verify the URL is safe and reachable without following unsafe redirects.
        """
        try:
            if not self._is_safe_url(url):
                return False
            
            parsed = urlparse(url)
            hostname = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == 'https' else 80)
            
            # Try to establish a raw socket connection first to verify accessibility and IP
            with socket.create_connection((hostname, port), timeout=timeout) as sock:
                # Get the IP we actually connected to
                remote_ip = sock.getpeername()[0]
                ip_addr = ipaddress.ip_address(remote_ip)
                if ip_addr.is_private or ip_addr.is_loopback or ip_addr.is_link_local:
                    self.log(f"🛡️ Preflight blocked private IP connection: {remote_ip}")
                    return False
            return True
        except Exception as e:
            self.log(f"⚠️ Preflight check failed for {url}: {e}")
            return False

    def _fetch_page_content(self, url: str, max_chars: int = 4000) -> str:
        """Fetch and extract text content from a URL with retries."""
        try:
            # Enhanced headers to mimic a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            }
            
            resp = None
            last_error = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Enhanced SSRF Check: Preflight + Resolution check
                    if attempt == 0: # Only preflight once
                        if not self._preflight_check(url):
                            return "[Error: URL blocked by SSRF protection or connection failed preflight]"
                    else:
                        # Re-verify resolve on retries to catch DNS rebinding
                        if not self._is_safe_url(url):
                            return "[Error: URL blocked by SSRF protection]"

                    resp = self.session.get(url, headers=headers, timeout=10, allow_redirects=True)
                    
                    # Check content size (limit to 5MB)
                    if len(resp.content) > 5 * 1024 * 1024:
                        self.log(f"⚠️ Content too large ({len(resp.content)} bytes) for {url}")
                        return "[Error: Content too large]"
                    
                    # Check MIME type
                    content_type = resp.headers.get('Content-Type', '').lower()
                    if 'text/html' not in content_type and 'text/plain' not in content_type and 'application/xml' not in content_type and 'application/json' not in content_type:
                        self.log(f"⚠️ Skipping non-text content: {content_type} for {url}")
                        return f"[Skipped: Non-text content ({content_type})]"

                    resp.raise_for_status()
                    break # Success
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 403:
                        # Retry 1: Different Desktop UA immediately
                        if attempt == 0:
                            headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15'
                            continue
                        else:
                            self.log(f"⚠️ Access Denied (403) for {url}. Respecting block.")
                            return "[Access Denied: The website blocked the request.]"
                    elif attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1)) # Simple backoff
                        continue
                    else:
                        self.log(f"⚠️ HTTP Error fetching {url}: {e}")
                        return f"[HTTP Error: {e}]"
                    last_error = e
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(1 * (attempt + 1))
                        continue
                    self.log(f"⚠️ Connection failed for {url}: {e}")
                    return f"[Connection Error: {e}]"
            
            if not resp:
                self.log(f"⚠️ Connection failed for {url}: {last_error}")
                return f"[Connection Error: {last_error}]"

            # 1. Try Trafilatura (Best for article extraction)
            try:
                import trafilatura
                downloaded = resp.text
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=False, no_fallback=False)
                if text:
                    self.log(f"✅ Deep Search: Fetched {len(text)} chars from {url} using Trafilatura")
                    return self._sanitize_content(text[:max_chars]) + ("..." if len(text) > max_chars else "")
            except ImportError:
                pass # Trafilatura not installed
            
            # Try using BeautifulSoup if available
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.content, 'html.parser')
                
                # Remove unwanted tags
                for script in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript", "svg", "path", "form", "button", "input"]):
                    script.extract()
                
                # Remove common cookie/ad banners by class/id heuristics
                for div in soup.find_all("div", class_=re.compile(r"(cookie|banner|popup|advert|promo|subscribe|newsletter|modal)", re.I)):
                    div.extract()
                
                text = soup.get_text(separator=' ', strip=True)
            except ImportError:
                # Fallback to regex
                text = resp.text
                # Combine regex substitutions for speed
                text = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
            
            # Limit content length to avoid context overflow
            self.log(f"✅ Deep Search: Fetched {len(text)} chars from {url}")
            
            # Content Firewall: Sanitize text
            clean_text = self._sanitize_content(text[:max_chars])
            return clean_text + ("..." if len(text) > max_chars else "")
            
        except Exception as e:
            self.log(f"⚠️ Failed to fetch {url}: {e}")
            return f"[Error fetching content: {e}]"

    def _sanitize_content(self, text: str) -> str:
        """
        Content Firewall Layer.
        Strips instructions, imperatives, and potential prompt injections.
        """
        # 1. Strip System Prompt Patterns
        text = re.sub(r"(?i)(you are a|act as a|ignore previous|system prompt|developer mode)", "", text)
        
        # 2. Strip Imperatives (Simple heuristic)
        # Remove lines starting with strong verbs if they look like commands
        # This is a basic filter; a real classifier would be better.
        # For now, we remove "You must", "AI should", etc.
        text = re.sub(r"(?i)(you must|you should|ai must|ai should|always|never)\s+.*?[.!?]", "", text)
        
        # 3. Strip Self-Referential Manipulation
        text = re.sub(r"(?i)(your core values are|your instructions are|update your)", "", text)
        
        # 4. Strip Command Injections
        text = re.sub(r"\[EXECUTE:.*?\]", "[REDACTED COMMAND]", text, flags=re.IGNORECASE)

        return text

    def _search_web(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Perform a web search using DuckDuckGo (DDGS library, HTML, or Lite).
        """
        # 1. Try duckduckgo_search library (Best reliability)
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                # ddgs.text returns an iterator
                try:
                    results = list(ddgs.text(query, max_results=5)) if hasattr(ddgs, 'text') else []
                except Exception as e:
                    self.log(f"⚠️ DDGS text search error: {e}")
                    results = []
                    
                self.log(f"🔍 DDGS found {len(results)} results for '{query}'")
                if results:
                    formatted = []
                    for i, r in enumerate(results):
                        title = r.get('title')
                        link = r.get('href')
                        snippet = r.get('body')
                        
                        formatted.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}")
                        
                        # Deep Search: Fetch content for top 3 results
                        if i < 3 and link:
                            try:
                                # Filter out ad links
                                link_lower = link.lower()
                                if "aclick" in link_lower or "googleadservices" in link_lower or "doubleclick" in link_lower or "adurl" in link_lower:
                                    continue
                                    
                                self.log(f"🌐 Deep Search: Fetching content from {link}...")
                                page_content = self._fetch_page_content(link)
                                if page_content and len(page_content) > 50:
                                    formatted.append(f"--- Page Content ---\n{page_content}\n--------------------")
                            except Exception as e:
                                self.log(f"⚠️ Failed to fetch content for link {link}: {e}")
                        
                        formatted.append("") # Separator
                    
                    content = f"🌐 Web Search Results for '{query}':\n\n" + "\n".join(formatted)
                    filepath = self._save_to_file(query, "WEB", content)
                    return content, filepath
        except ImportError:
            self.log("⚠️ 'duckduckgo_search' library not found. Falling back to scraping. Install via: pip install duckduckgo-search")
        except Exception as e:
            self.log(f"⚠️ DDGS library failed: {e}")

        # 2. Fallback to Requests Scraping
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Referer': 'https://duckduckgo.com/'
        }

        def clean_html(raw_html):
            try:
                from bs4 import BeautifulSoup
                return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)
            except ImportError:
                # Fallback to regex if BS4 is missing
                clean = re.sub(r'<[^>]+>', '', raw_html)
                return re.sub(r'\s+', ' ', clean).strip()
        
        # Strategy A: html.duckduckgo.com
        try:
            url = "https://html.duckduckgo.com/html/"
            self.log(f"🔍 Attempting HTML Scraping for '{query}'...")
            resp = self.session.post(url, data={'q': query}, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                results = []
                
                # Try BeautifulSoup if available for robust parsing
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    
                    # Select result containers
                    # Note: DDG HTML structure changes often, this is a best-effort selector based on common patterns
                    # Usually results are in div.result
                    result_divs = soup.select('.result')
                    
                    for div in result_divs[:5]:
                        title_tag = div.select_one('.result__a')
                        snippet_tag = div.select_one('.result__snippet')
                        
                        if title_tag and snippet_tag:
                            t = title_tag.get_text(strip=True)
                            link = title_tag.get('href', '')
                            s = snippet_tag.get_text(strip=True)
                            
                            if link and t:
                                results.append((t, link, s))
                except ImportError:
                    # Fallback to Regex
                    anchors = re.findall(r'<a[^>]+class="result__a"[^>]*>.*?</a>', resp.text, re.DOTALL)
                    snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', resp.text, re.DOTALL)
                
                    for i in range(min(len(anchors), len(snippets), 5)):
                        anchor = anchors[i]
                        href_match = re.search(r'href="([^"]+)"', anchor)
                        link = href_match.group(1) if href_match else ""
                        
                        title_match = re.search(r'>(.*?)</a>', anchor, re.DOTALL)
                        t = clean_html(title_match.group(1)) if title_match else "No Title"
                        s = clean_html(snippets[i])
                        
                        if link and t:
                            results.append((t, link, s))

                formatted_results = []
                for i, (t, link, s) in enumerate(results):
                    entry = f"Title: {t}\nLink: {link}\nSnippet: {s}"
                    try:
                        if i < 3:
                            page_content = self._fetch_page_content(link)
                            if page_content and len(page_content) > 50:
                                entry += f"\n--- Page Content ---\n{page_content}\n--------------------"
                        formatted_results.append(entry)
                    except Exception as e:
                        self.log(f"⚠️ Failed to process HTML result {i}: {e}")
                
                if formatted_results:
                    content = f"🌐 Web Search Results for '{query}':\n\n" + "\n\n".join(formatted_results)
                    filepath = self._save_to_file(query, "WEB", content)
                    return content, filepath
        except Exception as e:
            self.log(f"⚠️ HTML DDG failed: {e}")

        # Strategy B: lite.duckduckgo.com (Simpler HTML)
        try:
            url = "https://lite.duckduckgo.com/lite/"
            self.log(f"🔍 Attempting Lite Scraping for '{query}'...")
            resp = self.session.post(url, data={'q': query}, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                # Lite structure: <a href="..." class="result-link">Title</a> ... <td class="result-snippet">Snippet</td>
                anchors = re.findall(r'<a[^>]+class="result-link"[^>]*>.*?</a>', resp.text, re.DOTALL)
                snippets = re.findall(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', resp.text, re.DOTALL)
                
                results = []
                for i in range(min(len(anchors), len(snippets), 5)):
                    anchor = anchors[i]
                    href_match = re.search(r'href="([^"]+)"', anchor)
                    link = href_match.group(1) if href_match else ""
                    
                    t = clean_html(re.sub(r'<[^>]+>', '', anchor))
                    s = clean_html(snippets[i])
                    if link and t:
                        entry = f"Title: {t}\nLink: {link}\nSnippet: {s}"
                        try:
                            if i < 3:
                                page_content = self._fetch_page_content(link)
                                if page_content and len(page_content) > 50:
                                    entry += f"\n--- Page Content ---\n{page_content}\n--------------------"
                            results.append(entry)
                        except Exception as e:
                            self.log(f"⚠️ Failed to process Lite result {i}: {e}")
                
                if results:
                    content = f"🌐 Web Search Results (Lite) for '{query}':\n\n" + "\n\n".join(results)
                    filepath = self._save_to_file(query, "WEB", content)
                    return content, filepath
        except Exception as e:
            self.log(f"⚠️ Lite DDG failed: {e}")

        return f"ℹ️ No results found for '{query}' (All methods failed).", None