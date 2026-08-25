import csv
from pathlib import Path
from mcp_server.server import mcp

@mcp.tool()
def find_icd10_code(diagnosis_description: str) -> str:
    """
    Find the ICD-10-CM classification code for a diagnosis description.
    Use when finalizing the structured report codes section.
    
    NOTE: In production this would call a licensed ICD-10 API
    (e.g. CMS ICD-10-CM API or a commercial provider).
    Mock CSV used for portfolio demonstration purposes.
    """
    mock_csv = Path(__file__).parent.parent.parent / "data" / "mock" / "icd10_codes.csv"
    
    if not mock_csv.exists():
        return f"Error: Mock database not found at {mock_csv}"
        
    query = diagnosis_description.lower()
    best_match = None
    
    with open(mock_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            desc = row['description'].lower()
            # Simple substring matching for mock purposes
            if query in desc or desc in query:
                best_match = row
                break
            
            # Fallback to word overlap, ignoring short/common words
            query_words = {w for w in query.split() if len(w) > 4 and w not in ["disease", "syndrome", "disorder"]}
            desc_words = {w for w in desc.split() if len(w) > 4 and w not in ["disease", "syndrome", "disorder"]}
            if len(query_words.intersection(desc_words)) > 0:
                if not best_match:
                    best_match = row
                
    if best_match:
        return f"ICD-10 Code: {best_match['code']} | Description: {best_match['description']} | Category: {best_match['category']}"
        
    return f"No ICD-10 code found matching '{diagnosis_description}'. Please consult a certified medical coder."
