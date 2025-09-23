import os, re
from dotenv import load_dotenv
import requests
from lxml import etree
from colorama import Fore

load_dotenv()

api_key = os.getenv("elsevier_api_key")
FAILED_FILE = "faileddownloads.txt"
SUCCESS_FILE = "successfuldownloads.txt"

failed_color = Fore.RED
success_color = Fore.GREEN

def download_paper(api_key, DOI, full=True):
    
    view = "FULL" if full is True else ""
    
    # header with API key (can be put in url but this is easier to read)
    headers = { "X-ELS-APIKey":api_key }
    
    # url with DOI and view
    url = f"https://api.elsevier.com/content/article/doi/{DOI}?view={view}"
    
    try:
        res = requests.get(url, headers=headers)
        
    except requests.RequestException as e:
        print(failed_color + f"{DOI} -> request error: {e}")
        log(DOI, "RequestException", "fail")
        return -1
    
    if res.status_code != 200:
        print(failed_color + f"{DOI} -> status {res.status_code}")
        log(DOI, "status_code != 200", "fail")
        return -1

    try:
        # parse XML for title 
        root = etree.fromstring(res.content)
        titles = root.xpath(
            "//dc:title/text()", 
            namespaces={"dc": "http://purl.org/dc/elements/1.1/"}
        )
        
        output_filename = safe_filename(DOI)

        os.makedirs("downloads", exist_ok=True)  # make sure downloads folder exists
        
        with open(f"downloads/{trim_string(output_filename)}.xml", "wb") as f:
            try: 
                
                f.write(res.content)
                print(success_color + f"{DOI} -> status {res.status_code}")
                log(DOI, "success", "success")
                
            except Exception as e:
                print(failed_color + f"{DOI} -> save error")

    except Exception as e:
        print(failed_color + f"{DOI} -> parsing error: {e}")
        log(DOI, "parse/save", "fail")
        return -1

        
    return 1

def trim_string(s, max_len=200):
    return s if len(s) <= max_len else s[:max_len]

def log(DOI, fail_point, type):
    
    if type == "fail":
        with open(FAILED_FILE, "a") as f:
            f.write(f"{DOI} failed at {fail_point} \n")
    if type == "success":
        with open(SUCCESS_FILE, "a") as f:
            f.write(f"{DOI} failed at {fail_point} \n")
        

def load_dois(file_path="doi.txt"):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
def safe_filename(s: str, maxlen: int = 150) -> str:
    # Replace path separators and other problematic chars
    s = re.sub(r'[\\/*?:"<>|]', "_", s)
    # Collapse whitespace and trim
    s = re.sub(r"\s+", " ", s).strip()
    # Limit length so we don't hit OS limits
    return s[:maxlen] or "untitled"


DOIs = load_dois("doi.txt")

DOIs = list(set(DOIs))

print(Fore.BLUE + f"\nAttempting to download {len(DOIs)} DOIs")

#for DOI in DOIs:
    #download_paper(api_key=api_key, DOI=DOI, full=True)
