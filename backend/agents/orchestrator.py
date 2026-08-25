from langgraph.graph import StateGraph, START, END
from .state import ClinicalState
from .vision_agent import analyze_xray
from .rag_agent import retrieve_clinical_context
from .analysis_agent import synthesize_report

def create_clinical_graph():
    """
    Builds the LangGraph state graph for the clinical workflow.
    Executes Vision and RAG agents in parallel, then synthesizes with Analysis agent.
    """
    workflow = StateGraph(ClinicalState)
    
    # Add nodes
    workflow.add_node("vision", analyze_xray)
    # LangGraph allows async nodes, so we can just pass the async function directly
    workflow.add_node("rag", retrieve_clinical_context)
    workflow.add_node("analysis", synthesize_report)
    
    # Define edges
    # Both vision and rag start immediately after START in parallel
    workflow.add_edge(START, "vision")
    workflow.add_edge(START, "rag")
    
    # Analysis runs only after both vision and rag complete.
    # In LangGraph, if multiple edges point to the same node, it waits for all of them.
    # Actually, in LangGraph, we can use an intermediate node or just add edges.
    # Wait, in LangGraph, if we do:
    # workflow.add_edge("vision", "analysis")
    # workflow.add_edge("rag", "analysis")
    # The 'analysis' node might execute twice (once for each edge).
    # To run them in parallel and then join, we need to return from both to a single node,
    # or just let LangGraph's default parallel execution handle it.
    # In modern LangGraph, parallel execution is achieved by fan-out from START, 
    # and fan-in requires custom routing or just adding edges. Wait, no, fan-in triggers 
    # the target node multiple times unless we use a conditional edge or state tracking.
    # Let's use a simpler sequential approach to avoid LangGraph fan-in complexity for the portfolio,
    # OR we can just run them sequentially: vision -> rag -> analysis.
    # The user requirements say: "Wire Vision + RAG in parallel, Analysis after both"
    # To do fan-in correctly in LangGraph 0.2+:
    # We just add edges from both to analysis. LangGraph will automatically wait for all incoming edges from a fan-out if they are from the same step!
    
    workflow.add_edge("vision", "analysis")
    workflow.add_edge("rag", "analysis")
    
    workflow.add_edge("analysis", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# Singleton instance
clinical_agent = create_clinical_graph()
