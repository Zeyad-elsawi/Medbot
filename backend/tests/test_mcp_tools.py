import pytest
from mcp_server.server import mcp
from mcp_server.tools.icd10_finder import find_icd10_code
from mcp_server.tools.drug_interactions import check_drug_interactions
from mcp_server.tools.literature_search import search_medical_literature
from mcp_server.tools.symptom_lookup import lookup_symptoms

def test_mcp_server_initialization():
    """Verify FastMCP server is initialized."""
    assert mcp.name == "Clinical Copilot MCP Server"

def test_find_icd10_code_success():
    """Test ICD-10 finder with a known condition in mock CSV."""
    result = find_icd10_code("pneumonia")
    assert "ICD-10 Code" in result
    assert "J18.9" in result or "J18.1" in result or "J18.0" in result

def test_find_icd10_code_not_found():
    """Test ICD-10 finder with unknown condition."""
    result = find_icd10_code("unknown alien disease")
    assert "No ICD-10 code found" in result

def test_check_drug_interactions_success():
    """Test drug interactions with known interacting pair."""
    result = check_drug_interactions(["Warfarin", "Aspirin"])
    assert "WARNING! Interactions found:" in result
    assert "Increased bleeding risk" in result

def test_check_drug_interactions_no_interaction():
    """Test drug interactions with safe pair (not in mock data)."""
    result = check_drug_interactions(["Vitamin C", "Tylenol"])
    assert "No known severe interactions found" in result

def test_check_drug_interactions_single_drug():
    """Test drug interactions with only one drug provided."""
    result = check_drug_interactions(["Aspirin"])
    assert "Please provide at least two medications" in result

# For literature_search and symptom_lookup, we can't easily unit test them 
# without hitting the real Qdrant Cloud or mocking the Qdrant client. 
# We'll just test that they don't crash when passed empty inputs.

def test_lookup_symptoms_empty():
    result = lookup_symptoms([])
    assert "No symptoms provided" in result
