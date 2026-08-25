import logging
import sys
from pathlib import Path
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from .state import ClinicalState

logger = logging.getLogger(__name__)

async def extract_text_from_result(result) -> str:
    """Extracts text from a CallToolResult or returns string fallback."""
    if hasattr(result, "content") and result.content:
        return "\n".join(item.text for item in result.content if item.type == "text")
    return str(result)

async def retrieve_clinical_context(state: ClinicalState) -> dict:
    """
    RAG & Tools node for LangGraph.
    Acts as an MCP Client. Connects to our FastMCP server over STDIO and 
    programmatically calls the tools to gather medical context.
    """
    symptoms = state.get("symptoms_text", "")
    medications = state.get("current_medications", [])
    
    if not symptoms:
        logger.warning("No symptoms provided. Skipping RAG retrieval.")
        return {}
        
    logger.info("Initializing MCP Client to communicate with FastMCP server...")
    
    # We use the fastmcp executable directly from the same env
    fastmcp_exe = str(Path(sys.executable).parent / "fastmcp")
    server_path = str(Path(__file__).parent.parent / "mcp_server" / "server.py")
    
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent)
    
    server_params = StdioServerParameters(
        command=fastmcp_exe,
        args=["run", f"{server_path}:mcp"],
        env=env
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                logger.info("Initializing MCP Session...")
                await session.initialize()
                
                logger.info("Calling MCP Tools...")
                
                # 1. Lookup Symptoms
                res_symp = await session.call_tool(
                    "lookup_symptoms", 
                    arguments={"symptom_list": [s.strip() for s in symptoms.split(",")]}
                )
                symptom_matches = await extract_text_from_result(res_symp)
                
                # 2. Search Medical Literature
                res_lit = await session.call_tool(
                    "search_medical_literature", 
                    arguments={"query": symptoms, "top_k": 3}
                )
                literature_refs = await extract_text_from_result(res_lit)
                
                # 3. Find ICD-10 Code
                res_icd = await session.call_tool(
                    "find_icd10_code", 
                    arguments={"diagnosis_description": symptoms}
                )
                icd10_codes = await extract_text_from_result(res_icd)
                
                # 4. Check Drug Interactions
                drug_interactions = ""
                if medications and len(medications) >= 2:
                    res_drug = await session.call_tool(
                        "check_drug_interactions", 
                        arguments={"medications": medications}
                    )
                    drug_interactions = await extract_text_from_result(res_drug)
                
                logger.info("Successfully retrieved medical context via MCP.")
                
                # Store the string results in state
                # Note: The state expects Lists for some of these, but we can store them as strings 
                # or wrap them in lists. We'll store them as single-item lists if the state dictates it.
                return {
                    "symptom_matches": [symptom_matches],
                    "literature_refs": [literature_refs],
                    "icd10_codes": [icd10_codes],
                    "drug_interactions": [drug_interactions] if drug_interactions else []
                }
                
    except Exception as e:
        logger.error(f"Error in RAG Agent (MCP Client): {e}")
        return {"error": f"MCP Client failed: {str(e)}"}
