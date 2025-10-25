import re
import os
from datetime import datetime
from colorama import init, Fore, Style
from downloader import download_elsevier, download_wiley, download_springer_nature
from utilis import load_dois

# Initialize colorama for cross-platform color support
init(autoreset=True)

# -------------------------
# 1. DOI prefix → downloader function map
# -------------------------
DOI_DISPATCH = {
    # Springer Nature family
    "10.1007": download_springer_nature,
    "10.1038": download_springer_nature,
    "10.1186": download_springer_nature,
    "10.1023": download_springer_nature,
    "10.1140": download_springer_nature,
    "10.1365": download_springer_nature,
    "10.1057": download_springer_nature,

    # Elsevier
    "10.1016": download_elsevier,

    # Wiley
    "10.1002": download_wiley,
}

# -------------------------
# 2. Helpers
# -------------------------
def extract_valid_doi(text: str):
    """Extract the first valid DOI (if any) from a line of text."""
    text = text.strip().lower()
    match = re.search(r"10\.\d{4,9}/[^\s]+", text)
    return match.group(0) if match else None


def get_prefix(doi: str):
    """Extract DOI prefix (e.g., 10.1016)."""
    match = re.match(r"(10\.\d+)", doi)
    return match.group(1) if match else None


# -------------------------
# 3. Logging utilities
# -------------------------
LOG_DIR = "logs"
SUCCESS_LOG = os.path.join(LOG_DIR, "success.log")
FAILURE_LOG = os.path.join(LOG_DIR, "failures.log")

os.makedirs(LOG_DIR, exist_ok=True)

def log_result(file_path: str, message: str):
    """Append a message to a log file with a timestamp."""
    with open(file_path, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")


# -------------------------
# 4. Main execution pipeline
# -------------------------
def dispatch_downloads(doi_file: str):
    """
    Load DOIs from file, deduplicate them, group by downloader,
    ask once per downloader whether to proceed, then download.
    """
    print(Fore.CYAN + f"[INFO] Loading DOIs from {doi_file} ...")
    raw_dois = load_dois(doi_file)

    parsed = set()
    for line in raw_dois:
        doi = extract_valid_doi(line)
        if doi:
            parsed.add(doi)

    print(Fore.CYAN + f"[INFO] Found {len(parsed)} unique valid DOIs")

    # Group DOIs by downloader function
    grouped_by_downloader = {}
    for doi in parsed:
        prefix = get_prefix(doi)
        downloader = DOI_DISPATCH.get(prefix)
        if downloader:
            grouped_by_downloader.setdefault(downloader, []).append(doi)
        else:
            log_result(FAILURE_LOG, f"[WARN] Unsupported prefix {prefix} for DOI {doi}")

    # Interactive confirmation for each downloader
    selected_downloaders = []
    for downloader, dois in grouped_by_downloader.items():
        downloader_name = downloader.__name__
        user_input = input(
            Fore.MAGENTA
            + f"\n[INPUT] Download all {len(dois)} DOIs using {downloader_name}? [Y/n]: "
        ).strip().lower()

        if user_input in ("", "y", "yes"):
            selected_downloaders.append(downloader)
            print(Fore.GREEN + f"[SELECTED] Will use {downloader_name}")
        else:
            print(Fore.YELLOW + f"[SKIP] Skipping {downloader_name}")

    # Confirm before proceeding
    if not selected_downloaders:
        print(Fore.YELLOW + "[CANCELLED] No downloaders selected. Exiting.")
        return

    print(Style.BRIGHT + Fore.CYAN + "\n[INFO] You selected:")
    for d in selected_downloaders:
        print(Fore.GREEN + f"  • {d.__name__} ({len(grouped_by_downloader[d])} DOIs)")
    print()

    proceed = input(Fore.MAGENTA + "[INPUT] Proceed with all selected downloads? [Y/n]: ").strip().lower()
    if proceed not in ("", "y", "yes"):
        print(Fore.YELLOW + "[CANCELLED] Download process aborted.")
        return

    # Execute all selected downloads
    for downloader in selected_downloaders:
        dois = grouped_by_downloader[downloader]
        print(Fore.CYAN + f"\n[INFO] Starting downloads for {len(dois)} DOIs using {downloader.__name__} ...")

        for doi in sorted(dois):
            try:
                downloader(doi)
                msg = f"SUCCESS: {doi} via {downloader.__name__}"
                print(Fore.GREEN + f"[SUCCESS] {doi}")
                log_result(SUCCESS_LOG, msg)
            except Exception as e:
                err = f"FAIL: {doi} ({e})"
                print(Fore.RED + f"[ERROR] {doi} -> {e}")
                log_result(FAILURE_LOG, err)

    print(Style.BRIGHT + Fore.CYAN + "\n[INFO] Download process complete.")
    print(Fore.GREEN + f"✔ Successes logged to {SUCCESS_LOG}")
    print(Fore.RED + f"✘ Failures logged to {FAILURE_LOG}")


# -------------------------
# 5. CLI entry point
# -------------------------
if __name__ == "__main__":
    dispatch_downloads("doi.txt")
