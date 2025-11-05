import requests
import json
import os
import time

def harvest_crossref_cursor_multi(
    queries,
    rows=1000,
    max_records_per_query=None,  # None = keep going until exhausted
    outfile="polymer_metadata_cleaned.json",
    statsfile="polymer_query_stats.json",
    email="castro.31@wright.edu",
    save_every=5000
):
    """
    Harvest Crossref metadata using cursor-based pagination for multiple queries.
    Keeps only key fields, deduplicates by DOI, and streams results to disk.
    Simultaneously tracks per-query counts in a separate JSON stats file.
    """

    headers = {"User-Agent": f"PolymerHarvester/2.1 (mailto:{email})"}
    base_url = "https://api.crossref.org/works"

    # Initialize storage
    all_metadata = []
    seen_dois = set()
    query_stats = {}

    # Resume if existing files are found
    if os.path.exists(outfile):
        print(f"Resuming from existing {outfile}...")
        with open(outfile, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
        seen_dois = {entry.get("DOI") for entry in all_metadata if entry.get("DOI")}
        print(f"Loaded {len(seen_dois)} previously saved records.\n")

    if os.path.exists(statsfile):
        print(f"Resuming from {statsfile}...")
        with open(statsfile, "r", encoding="utf-8") as f:
            query_stats = json.load(f)
        print(f"Loaded stats for {len(query_stats)} queries.\n")

    # Process each search term
    for query in queries:
        print(f"\n🔍 Starting query: '{query}'")
        cursor = "*"
        fetched = 0
        retries = 0
        query_stats.setdefault(query, {"total_records": 0, "unique_records": 0})

        while True:
            params = {
                "query": query,
                "rows": rows,
                "cursor": cursor,
                "filter": "type:journal-article"
            }

            try:
                resp = requests.get(base_url, params=params, headers=headers, timeout=120)
                if resp.status_code != 200:
                    print(f"⚠️ HTTP {resp.status_code}. Retrying in 30s...")
                    time.sleep(30)
                    retries += 1
                    if retries > 3:
                        break
                    continue

                data = resp.json().get("message", {})
                items = data.get("items", [])
                if not items:
                    print("No more items for this query.")
                    break

                new_items = 0
                for item in items:
                    doi = item.get("DOI")
                    if not doi or doi in seen_dois:
                        continue
                    seen_dois.add(doi)

                    cleaned_item = {
                        "DOI": doi,
                        "title": item.get("title", [""])[0] if item.get("title") else "",
                        "publisher": item.get("publisher", ""),
                        "journal": item.get("container-title", [""])[0] if item.get("container-title") else "",
                        "publication_date": item.get("issued", {}).get("date-parts", [[None]])[0],
                        "link": item.get("link", [])
                    }

                    all_metadata.append(cleaned_item)
                    new_items += 1

                fetched += len(items)
                cursor = data.get("next-cursor")

                # Update statistics
                query_stats[query]["total_records"] += len(items)
                query_stats[query]["unique_records"] += new_items

                print(f"Fetched {fetched:,} records for '{query}' "
                      f"({new_items:,} new unique; total unique {len(seen_dois):,}).")

                # Periodic save of both metadata and stats
                if len(all_metadata) % save_every < rows or not cursor:
                    with open(outfile, "w", encoding="utf-8") as f:
                        json.dump(all_metadata, f, indent=2)
                    with open(statsfile, "w", encoding="utf-8") as f:
                        json.dump(query_stats, f, indent=2)
                    print(f"💾 Progress saved: {len(all_metadata):,} records | "
                          f"Stats for {len(query_stats):,} queries\n")

                # Exit if limit reached
                if max_records_per_query and fetched >= max_records_per_query:
                    print(f"Reached {max_records_per_query:,} record cap for '{query}'.")
                    break

                # Stop if no more pages
                if not cursor:
                    break

                time.sleep(1)  # rate control

            except Exception as e:
                print(f"⚠️ Error: {e}. Retrying in 60s...")
                time.sleep(60)
                continue

        # Save after each query completes
        with open(statsfile, "w", encoding="utf-8") as f:
            json.dump(query_stats, f, indent=2)
        print(f"📊 Query '{query}' complete: "
              f"{query_stats[query]['unique_records']:,} unique / "
              f"{query_stats[query]['total_records']:,} total.\n")

    # Final save
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)
    with open(statsfile, "w", encoding="utf-8") as f:
        json.dump(query_stats, f, indent=2)

    print(f"\n✅ Harvest complete.")
    print(f"📦 {len(all_metadata):,} total unique records saved to {outfile}.")
    print(f"📈 Per-query stats saved to {statsfile}.")


if __name__ == "__main__":
    polymer_queries = [
        "polymer",
        "polymer science",
        "polymer chemistry",
        "polymer physics",
        "polymer synthesis",
        "polymer composites",
        "polymer nanocomposites",
        "biodegradable polymers",
        "polymer materials",
        "polymer engineering",
        "polymer mechanics",
        "polymerization",
        "polymerize",
        "polymerized",
        "polymerizing",
        "conductive polymers",
        "smart polymers",
        "photoresponsive polymers",
        "polymer blends",
        "block copolymers",
        "thermoplastic polymers",
        "thermosetting polymers",
        "bioinspired polymers",
        "self-healing polymers",
        "biopolymers",
        "polymer coatings",
        "nanostructured polymers",
        "crosslinked polymers",
        "recyclable polymers"
    ]

    harvest_crossref_cursor_multi(
        queries=polymer_queries,
        rows=1000,
        max_records_per_query=None,  # unlimited
        outfile="polymer_metadata_cleaned.json",
        statsfile="polymer_query_stats.json"
    )
