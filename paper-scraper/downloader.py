'''

This file contains (will contain, not yet complete) several calls to
APIs of scientific publishers to download papers from scientific publishers

Completed:
- Elsevier (10.1016) https://dev.elsevier.com/

Will likely be able to support:

- Springer / Springer Nature (10.1007, 10.1038) - in progress https://datasolutions.springernature.com/account/api-management/
- IEEE (10.1109)
- IOP Publishing (10.1088)
- AIP (10.1063)
- APS / Physical Review (10.1103)
- ACS (10.1021)
- Wiley (10.1002) - in progress, needs testing! https://github.com/WileyLabs/tdm-client?tab=readme-ov-file#wiley-tdm-client
- MDPI (10.3390)

UM IDK???:
- SAGE 
    1 request every 6 seconds - Monday to Friday between Midnight and Noon in the "America/Los_Angeles" timezone;
    1 request every 2 seconds - Monday to Friday between Noon and Midnight in the "America/Los_Angeles" timezone, and all day Saturday and Sunday.

Will likely not be able to support:
- World Scientific (10.1142)
- Cambridge / Materials Research Society (10.1557)
- Taylor & Francis (10.1080)
- Royal Society of Chemistry (10.1039) why do I have an API key for this???????? (ok I think the API is )
- AIAA (10.2514)
- Smaller / niche publishers (e.g., 10.3144)

'''

import os
from dotenv import load_dotenv
import requests
from utilis import save_contents_to_file
from wiley_tdm import TDMClient
from requests import RequestException

class DownloadError(Exception):
    def __init__(self, doi, source, error_type, message):
        super().__init__(f"[{source}] {doi} - {error_type}: {message}")
        self.doi = doi
        self.source = source
        self.error_type = error_type
        self.message = message

    def __str__(self):
        return f"[{self.source}] {self.doi} - {self.error_type}: {self.message}"

load_dotenv()
elsevier_api_key = os.getenv("elsevier_api_key")
wiley_api_key = os.getenv("wiley_api_key")
springer_nature_api_key = os.getenv("springer_nature_open_access_api_key")

wiley_tdm = TDMClient(api_token=wiley_api_key, download_dir="downloads")

def download_springer_nature(DOI):
    url = "https://api.springernature.com/openaccess/jats"

    params = {
        "q": f"doi:{DOI}",
        "api_key": springer_nature_api_key
    }

    headers = {
        "Accept": "application/jats+xml"
    }
    
    try:
        res = requests.get(url, headers=headers, params=params, timeout=20)
        
        print(res)
        
        if res.status_code != 200:
            raise DownloadError(DOI, "springer_nature", type(e).__name__, "res.status_code != 200", str(e))
        
        paper_contents = res.content
        try:
            save_contents_to_file(DOI, paper_contents)
            
        except Exception as e:
           raise DownloadError(DOI, "springer_nature", type(e).__name__, "save error", str(e))
        
    except RequestException as e:
        raise DownloadError(DOI, "springer_nature", type(e).__name__, "request error", str(e))

    return

def download_wiley(DOI):
    
    try: 
        result = wiley_tdm.download_pdf(DOI)
        if not result or "Api Error" in str(result) or "Invalid" in str(result):
            raise DownloadError(DOI, "wiley", "APIError", str(result))

    except Exception as e:
        raise DownloadError(DOI, "wiley", type(e).__name__, str(e))

def download_elsevier(DOI):
    
    # header with API key (can be put in url but this is easier to read and probably better)
    headers = { "X-ELS-APIKey": elsevier_api_key }
    
    # url with DOI and download mode
    url = f"https://api.elsevier.com/content/article/doi/{DOI}?view=FULL"
    
    # attempt request
    try:
        res = requests.get(url, headers=headers)
        
        # check for status code error on successful requests
        if res.status_code != 200:
            raise DownloadError(DOI, "elsevier", "HTTPError", f"Status code {res.status_code}")
            
        paper_contents = res.content
        try:
            save_contents_to_file(DOI, paper_contents)
            
        except Exception as e:
           raise DownloadError(DOI, "elsevier", type(e).__name__, "save error", str(e))
 
    # catch error
    except requests.exceptions.RequestException as e:
        raise DownloadError(DOI, "elsevier", type(e).__name__, str(e))