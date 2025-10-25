'''

This file contains (will contain, not yet complete) several calls to
APIs of scientific publishers to download polymer-related papers from 
the provied dataset of DOIs.

Completed:
- Elsevier (10.1016)

Will likely be able to support:

- Springer / Springer Nature (10.1007, 10.1038)
- IEEE (10.1109)
- IOP Publishing (10.1088)
- AIP (10.1063)
- APS / Physical Review (10.1103)
- ACS (10.1021)
- Wiley (10.1002)
- MDPI (10.3390)

UM IDK???:
- SAGE 
    1 request every 6 seconds - Monday to Friday between Midnight and Noon in the "America/Los_Angeles" timezone;
    1 request every 2 seconds - Monday to Friday between Noon and Midnight in the "America/Los_Angeles" timezone, and all day Saturday and Sunday.

Will likely not be able to support:
- SAGE (10.1177) ()
- World Scientific (10.1142)
- Cambridge / Materials Research Society (10.1557)
- Taylor & Francis (10.1080)
- Royal Society of Chemistry (10.1039)
- AIAA (10.2514)
- Smaller / niche publishers (e.g., 10.3144)

'''

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

def download_pipeline():

    DOIs = load_dois("doi.txt")

    DOIs = list(set(DOIs))

    print(Fore.BLUE + f"\nAttempting to download {len(DOIs)} DOIs")

#for DOI in DOIs:
    #download_paper(api_key=api_key, DOI=DOI, full=True)
