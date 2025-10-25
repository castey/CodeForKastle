import re, os
from colorama import Fore
failed_color = Fore.RED
success_color = Fore.GREEN

def load_dois(file_path="doi.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
def trim_string(s, max_len=200):
    return s if len(s) <= max_len else s[:max_len]

def safe_filename(s: str, maxlen: int = 150) -> str:
    # Replace path separators and other problematic chars
    s = re.sub(r'[\\/*?:"<>|]', "_", s)
    # Collapse whitespace and trim
    s = re.sub(r"\s+", " ", s).strip()
    # Limit length so we don't hit OS limits
    return s[:maxlen] or "untitled"


def save_contents_to_file(DOI, contents):
    # parse filename for save downlaod
    try:
        output_filename = safe_filename(DOI)
        os.makedirs("downloads", exist_ok=True)
    
    except Exception as e:
        print(failed_color + f"{DOI} -> parsing error: {e}")
        return -1
        
    with open(f"downloads/{(output_filename)}.xml", "wb") as f:
        try: 
            f.write(contents)
            print(success_color + f"{DOI} -> save success!")

        except Exception as e:
            print(failed_color + f"{DOI} -> save error!")