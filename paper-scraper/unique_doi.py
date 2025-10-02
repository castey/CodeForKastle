from main import load_dois
import re

DOIs = load_dois("doi.txt")
parsed_DOIs = []

for DOI in DOIs:
    parsed_DOI = re.findall(r"\d{2}\..+", DOI)
    
    if len(parsed_DOI) > 0:
        parsed_DOIs.append(parsed_DOI[0])
    
cleaned_prefixes_final = []

print(len(list(set(parsed_DOIs))))

for D in parsed_DOIs:
    
    prefix = re.findall(r"\d{2}\.\d+\/", D)
    
    prefix = prefix[0]
    prefix = prefix[:-1]
    
    cleaned_prefixes_final.append(prefix)

unique_prefixes = list(set(cleaned_prefixes_final))

print(unique_prefixes)

