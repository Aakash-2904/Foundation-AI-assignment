import re
import time
import requests
from typing import Optional


REMOTIVE_API  = "https://remotive.com/api/remote-jobs"
ARBEITNOW_API = "https://www.arbeitnow.com/api/job-board-api"


def fetch_remotive_jobs(
    search: Optional[str] = None,
    category: str = "machine-learning",
    limit: int = 5,
) -> list[dict]:
# I used the remotive pblic API here defaulting category to 'ML'
# # mainly focusing on ML roles, returns job list
    params = {"limit": limit, "category": category}
    if search:
        params["search"] = search

    try:
        r = requests.get(REMOTIVE_API, params=params, timeout=12)
        r.raise_for_status()
        jobs = r.json().get("jobs", [])
    except requests.RequestException as e:
        print(f"[Remotive] Request failed: {e}")
        return [] 

    return [
        {
            "source":      "remotive",
            "title":       j.get("title", ""),
            "company":     j.get("company_name", ""),
            "location":    j.get("candidate_required_location", "Remote"),
            "tags":        j.get("tags", []),
            "description": _strip_html(j.get("description", "")),
            "url":         j.get("url", ""),
        }
        for j in jobs
    ]


def fetch_arbeitnow_jobs(
    search: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
## also using arbeitnow to get more jobs
    params = {}
    if search:
        params["q"] = search

    try:
        r = requests.get(ARBEITNOW_API, params=params, timeout=12)
        r.raise_for_status()
        jobs = r.json().get("data", [])
    except requests.RequestException as e:
        print(f"[Arbeitnow] Request failed: {e}")
        return []

    return [
        {
            "source":      "arbeitnow",
            "title":       j.get("title", ""),
            "company":     j.get("company_name", ""),
            "location":    j.get("location", "Remote"),
            "tags":        j.get("tags", []),
            "description": _strip_html(j.get("description", "")),
            "url":         j.get("url", ""),
        }
        for j in jobs[:limit]
    ]


def fetch_all_jobs(
    search: str = "machine learning engineer",
    limit_per_source: int = 5,
) -> list[dict]:
    # I added small sleep to avoid too many requests
    print(f"\n[Scraper] Fetching Remotive  jobs  (search='{search}') ...")
    remotive  = fetch_remotive_jobs(search=search, limit=limit_per_source)
    time.sleep(0.5)

    print(f"[Scraper] Fetching Arbeitnow jobs  (search='{search}') ...")
    arbeitnow = fetch_arbeitnow_jobs(search=search, limit=limit_per_source)

    all_jobs = remotive + arbeitnow
    print(f"[Scraper] Total jobs collected: {len(all_jobs)}\n")
    return all_jobs


def _strip_html(html: str) -> str:
    # used regex instead of bs4 (html is simple anyway)
    # full parser felt unnecessary, also cleaning html chars
    text = re.sub(r"<[^>]+>", " ", html)
    for ent, repl in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&#39;", "'")]:
        text = text.replace(ent, repl)
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    jobs = fetch_all_jobs(search="deep learning", limit_per_source=3)
    for i, job in enumerate(jobs, 1):
        print(f"[{i}] {job['title']} @ {job['company']}  ({job['source']})")
        print(f"     {job['description'][:120]}...\n")
