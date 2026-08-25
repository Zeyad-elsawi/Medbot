import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage
from agents.orchestrator import clinical_agent
from agents.state import ClinicalState

@pytest.mark.asyncio
@patch("agents.vision_agent.ChatOllama.invoke")
@patch("agents.analysis_agent.ChatOllama.invoke")
async def test_clinical_orchestrator(mock_vision_invoke, mock_analysis_invoke):
    """
    Integration test for the full LangGraph workflow.
    Requires Qdrant and FastMCP to be functional. LLMs are mocked.
    """
    # Setup mocks
    mock_vision_invoke.return_value = AIMessage(content="Structured Findings: Pleural effusion detected. No pneumothorax.")
    mock_analysis_invoke.return_value = AIMessage(content="""
1. **Summary of Findings**: Pleural effusion.
2. **Differential Diagnosis**: Infection, Heart Failure.
3. **Clinical Codes**: J90.
4. **Safety & Interactions**: Monitor Warfarin + Aspirin.
5. **Next Steps**: Thoracentesis.

*Disclaimer: AI-generated report for research demonstration. Not medical advice.*
""")

    initial_state: ClinicalState = {
        "xray_image_path": "data/openi/images/CXR83_IM-2358-1001.png",
        "symptoms_text": "chest pain, shortness of breath, chronic cough",
        "current_medications": ["Aspirin", "Warfarin"],
        "patient_age": 65,
        "patient_sex": "M"
    }
    
    # Run the compiled LangGraph app
    final_state = await clinical_agent.ainvoke(initial_state)
    
    # Verify Vision Agent output
    assert "xray_findings" in final_state
    assert len(final_state["xray_findings"]) > 0
    
    # Verify RAG Agent output (from MCP tools)
    assert "symptom_matches" in final_state
    assert "icd10_codes" in final_state
    assert "drug_interactions" in final_state
    assert "literature_refs" in final_state
    
    # Verify Analysis Agent output
    assert "final_report" in final_state
    assert "Disclaimer: AI-generated report" in final_state["final_report"]
