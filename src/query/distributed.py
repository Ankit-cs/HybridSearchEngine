import aiohttp
import asyncio
from src.ranking.fixed_point import sort_by_fixed_point

async def fetch_results(session: aiohttp.ClientSession, url: str, params: dict):
    """Fetch search results from a single worker node."""
    try:
        async with session.get(f"{url}/api/v1/search", params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("results", [])
            else:
                print(f"[Distributed] Worker {url} returned status {response.status}")
                return []
    except Exception as e:
        print(f"[Distributed] Failed to fetch from {url}: {e}")
        return []

async def scatter_gather_search(query: str, k: int, worker_urls: list[str], **kwargs):
    """
    Scatter a search query to multiple worker nodes and gather/merge the results.
    """
    params = {"q": query, "k": k}
    for key, val in kwargs.items():
        if val is not None:
            params[key] = val

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_results(session, url, params) for url in worker_urls]
        all_results_lists = await asyncio.gather(*tasks)

    # Merge results from all workers
    # We maintain a dictionary to deduplicate by doc_id (in case of overlaps)
    merged_scores = {}
    doc_metadata = {}

    for results in all_results_lists:
        for res in results:
            doc_id = res["doc_id"]
            score = res["score"]
            # Deduplicate, keeping the highest score
            if doc_id not in merged_scores or score > merged_scores[doc_id]:
                merged_scores[doc_id] = score
                doc_metadata[doc_id] = res

    # Sort merged results by fixed-point representation to maintain consistency
    ranked = sort_by_fixed_point(list(merged_scores.items()))

    # Reconstruct top-k final results
    final_results = []
    for doc_id, _ in ranked[:k]:
        final_results.append(doc_metadata[doc_id])

    return final_results
