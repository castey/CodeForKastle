import requests
import re

def get_crossref_metadata(doi: str, email: str = "castro.31@wright.edu") -> dict:
    """
    Query Crossref API for metadata of a given DOI.
    Returns the JSON metadata as a dictionary.
    """
    url = f"https://api.crossref.org/works/{doi}"
    headers = {"User-Agent": f"CrossrefFetcher/1.0 (mailto:{email})"}
    
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code == 200:
        return resp.json()
    else:
        raise Exception(f"Failed to fetch metadata: HTTP {resp.status_code}")


def sanitize_filename(name: str, max_len: int = 200) -> str:
    """
    Clean up a string to be a safe filename (alphanumeric + underscores).
    Trims to max_len characters.
    """
    # Replace spaces and forbidden characters
    safe = re.sub(r'[<>:"/\\|?*]', '', name)  
    safe = re.sub(r'\s+', '_', safe)  
    return safe[:max_len]


def download_pdf(metadata: dict) -> bool:
    """
    Takes Crossref metadata response and downloads the PDF if available.
    File is named after the paper title (trimmed to max 200 chars).
    Returns True if successful, False otherwise.
    """
    title_list = metadata.get("message", {}).get("title", [])
    if title_list:
        title = title_list[0]
    else:
        title = "paper"

    filename = sanitize_filename(title, max_len=200) + ".pdf"

    links = metadata.get("message", {}).get("link", [])
    pdf_url = None
    for l in links:
        if l.get("content-type") == "application/pdf":
            pdf_url = l.get("URL")
            break
    
    print(pdf_url)
    if not pdf_url:
        print("No PDF link found in metadata.")
        return False

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/117.0.0.0 Safari/537.36",
        "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://journals.sagepub.com/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive"
    }

    resp = requests.get(pdf_url, headers=headers, stream=True)

    if resp.status_code == 200 and "application/pdf" in resp.headers.get("Content-Type", ""):
        with open(filename, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
        print(f"PDF saved as {filename}")
        return True
    else:
        print(f"Could not download PDF. Status: {resp.status_code}")
        return False


# Example usage:
if __name__ == "__main__":
    doi = "10.1177/0954008315588983"
    metadata = get_crossref_metadata(doi)
    download_pdf(metadata)
