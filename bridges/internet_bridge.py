import requests
import time
import json
import os
from typing import Optional, Callable, Dict, Tuple
import xml.etree.ElementTree as ET

try:
    from lm import run_local_lm
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lm import run_local_lm

class InternetBridge:
    """
    Strictly limited internet access bridge.
    Allowed sources: WIKIPEDIA, PUBMED, ARXIV.
    Capabilities: Read-only search and summary retrieval.
    """
    ALLOWED_SOURCES = {"WIKIPEDIA", "PUBMED", "ARXIV"}

    def __init__(self, get_settings_fn: Callable[[], Dict], log_fn=print, save_dir: str = "./data/internet_data"):
        self.get_settings = get_settings_fn
        self.log = log_fn
        self.save_dir = save_dir
        self.last_request_time = 0
        self.request_interval = 2.0 # Rate limiting to be polite
        
        os.makedirs(self.save_dir, exist_ok=True)

    def _check_connectivity(self) -> bool:
        """Simple check to see if we can reach the internet."""
        try:
            # Try to reach a reliable host (Google DNS)
            requests.get("https://8.8.8.8", timeout=3)
            return True
        except Exception:
            return False

    def search(self, query: str, source: str) -> Tuple[str, Optional[str]]:
        """
        Execute an approved search request.
        Returns: (content_string, filepath_if_saved)
        """
        # 1. Rate Limit Check
        if time.time() - self.last_request_time < self.request_interval:
            return "⚠️ Rate limit active. Please wait a moment.", None
        self.last_request_time = time.time()

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

        self.log(f"🌐 Bridge executing search on {source}: {query}")

        try:
            if source == "WIKIPEDIA":
                return self._search_wikipedia(query)
            elif source == "PUBMED":
                return self._search_pubmed(query)
            elif source == "ARXIV":
                return self._search_arxiv(query)
            return "❌ Unknown source.", None
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
                messages=[{"role": "user", "content": "Check query."}],
                system_prompt=prompt,
                temperature=0.0,
                max_tokens=5,
                base_url=settings.get("base_url"),
                chat_model=settings.get("chat_model")
            )
            return "PASS" in response.upper()
        except Exception as e:
            self.log(f"⚠️ Safety check failed: {e}")
            return False # Fail closed

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
        
        resp = requests.get(url, params=params, headers=headers, timeout=10)
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
        
        resp_content = requests.get(url, params=params_content, headers=headers, timeout=10)
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
        
        resp = requests.get(search_url, params=search_params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return "ℹ️ No PubMed results found."
            
        # 2. ESummary (Get details)
        summary_url = f"{base_url}/esummary.fcgi"
        summary_params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "json"}
        
        resp_sum = requests.get(summary_url, params=summary_params, timeout=10)
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
        
        resp = requests.get(url, params=params, timeout=10)
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