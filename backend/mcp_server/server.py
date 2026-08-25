import os
from pathlib import Path
from fastmcp import FastMCP

# Initialize the FastMCP server
# The tools will be imported and registered below
mcp = FastMCP("Clinical Copilot MCP Server")

@mcp.resource("mcp://guidelines/{condition}")
def get_treatment_guidelines(condition: str) -> str:
    """
    Retrieve standard clinical treatment guidelines for a given condition.
    
    NOTE: This is a portfolio demonstration using mock markdown files.
    In a production setting, this would query a licensed clinical 
    guidelines database (e.g., UpToDate, BMJ Best Practice, NHS guidelines API).
    """
    # Look up the condition in the guidelines directory
    guidelines_dir = Path(__file__).parent / "guidelines"
    file_path = guidelines_dir / f"{condition.lower()}.md"

    if not file_path.exists():
        # Fallback to search if exact match not found
        available = [f.stem for f in guidelines_dir.glob("*.md")]
        return f"No specific guidelines found for condition: '{condition}'. Available guidelines: {', '.join(available)}"

    return file_path.read_text(encoding="utf-8")

from mcp_server.tools import literature_search, symptom_lookup, icd10_finder, drug_interactions
