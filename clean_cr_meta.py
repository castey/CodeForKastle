import json
import os

def clean_crossref_metadata(
    infile="polymer_metadata_offset.json",
    outfile="polymer_metadata_cleaned.json"
):
    """
    Cleans a Crossref JSON dump so that only key fields are preserved:
    DOI, title, publisher, container-title (journal), issued (publication date), link.
    """

    if not os.path.exists(infile):
        print(f"❌ Input file '{infile}' not found.")
        return

    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from {infile}...")

    cleaned = []
    for item in data:
        cleaned_item = {
            "DOI": item.get("DOI"),
            "title": item.get("title", [""])[0] if item.get("title") else "",
            "publisher": item.get("publisher", ""),
            "journal": item.get("container-title", [""])[0] if item.get("container-title") else "",
            "publication_date": item.get("issued", {}).get("date-parts", [[None]])[0],
            "link": item.get("link", [])
        }
        cleaned.append(cleaned_item)

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)

    print(f"✅ Cleaned {len(cleaned)} records and saved to {outfile}.")


if __name__ == "__main__":
    clean_crossref_metadata()
