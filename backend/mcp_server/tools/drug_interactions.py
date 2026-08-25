import csv
from typing import List
from pathlib import Path
from mcp_server.server import mcp

@mcp.tool()
def check_drug_interactions(medications: List[str]) -> str:
    """
    Check for known interactions between a list of medications.
    Use when patient history includes multiple medications.
    
    NOTE: In production this would call a licensed drug interaction API
    (e.g. FDA API or Micromedex).
    Mock CSV used for portfolio demonstration purposes.
    """
    if not medications or len(medications) < 2:
        return "No interactions to check. Please provide at least two medications."
        
    mock_csv = Path(__file__).parent.parent.parent / "data" / "mock" / "drug_interactions.csv"
    
    if not mock_csv.exists():
        return f"Error: Mock database not found at {mock_csv}"
        
    interactions_found = []
    meds_lower = [m.lower() for m in medications]
    
    with open(mock_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d1 = row['drug_1'].lower()
            d2 = row['drug_2'].lower()
            
            # Check if both drugs in the row are in the provided list
            if any(d1 in m or m in d1 for m in meds_lower) and any(d2 in m or m in d2 for m in meds_lower):
                interactions_found.append(
                    f"[Severity: {row['severity']}] {row['drug_1']} + {row['drug_2']}: {row['description']}. Recommendation: {row['recommendation']}"
                )
                
    if interactions_found:
        return "WARNING! Interactions found:\n- " + "\n- ".join(interactions_found)
        
    return "No known severe interactions found among the provided medications based on current mock database."
