"""
MLflow model for deploying memory-enabled agent to Model Serving.

Deploy this model to expose memory operations via ai_query().
"""

import mlflow
from mlflow.pyfunc import PythonModel
import os
from typing import Any, Dict, Union
import json


class MemoryAgentEndpoint(PythonModel):
    """
    MLflow model that exposes soul.py memory operations.
    
    Supports operations:
        - ask: Query with memory context
        - remember: Store a memory
        - recall: Search memories
        - decide: Log a decision
        - identity: Get/set agent identity
    """
    
    def load_context(self, context):
        """Load the memory agent with Delta backend."""
        from agents.memory.delta_backend import DeltaMemoryBackend
        from agents.memory.memory_agent import MemoryEnabledAgent
        
        # Get configuration from environment
        catalog = os.environ.get("SOUL_CATALOG", "main")
        schema = os.environ.get("SOUL_SCHEMA", "agent_memory")
        agent_id = os.environ.get("SOUL_AGENT_ID", "default")
        provider = os.environ.get("SOUL_PROVIDER", "anthropic")
        
        self.backend = DeltaMemoryBackend(
            catalog=catalog,
            schema=schema,
            agent_id=agent_id
        )
        
        self.agent = MemoryEnabledAgent(
            backend=self.backend,
            provider=provider
        )
    
    def predict(self, context, model_input: Union[Dict, str]) -> Dict[str, Any]:
        """
        Handle memory operations.
        
        Input format:
            {"operation": "ask|remember|recall|decide|identity", "query": "...", ...}
        
        Or just a string (treated as "ask" operation).
        """
        # Parse input
        if isinstance(model_input, str):
            try:
                model_input = json.loads(model_input)
            except json.JSONDecodeError:
                # Treat as simple ask query
                model_input = {"operation": "ask", "query": model_input}
        
        operation = model_input.get("operation", "ask")
        query = model_input.get("query", "")
        
        # Route to operation
        if operation == "ask":
            result = self.agent.ask(query)
            return {
                "answer": result["answer"],
                "route": result.get("route", "unknown"),
                "status": "success"
            }
        
        elif operation == "remember":
            role = model_input.get("role", "system")
            self.agent.remember(query, role=role)
            return {"status": "stored", "content": query}
        
        elif operation == "recall":
            k = model_input.get("k", 5)
            memories = self.agent.recall(query, k=k)
            return {"memories": memories, "count": len(memories), "status": "success"}
        
        elif operation == "decide":
            decision = model_input.get("decision", query)
            context_str = model_input.get("context", "")
            rationale = model_input.get("rationale", "")
            self.agent.decide(decision, context=context_str, rationale=rationale)
            return {"status": "decision_logged", "decision": decision}
        
        elif operation == "identity":
            if "content" in model_input:
                # Set identity
                self.agent.set_identity(model_input["content"])
                return {"status": "identity_updated"}
            else:
                # Get identity
                identity = self.agent.get_identity()
                return {"identity": identity, "status": "success"}
        
        elif operation == "history":
            limit = model_input.get("limit", 50)
            history = self.agent.get_recent_memory(limit=limit)
            return {"history": history, "status": "success"}
        
        elif operation == "decisions":
            limit = model_input.get("limit", 20)
            decisions = self.agent.get_decisions(limit=limit)
            return {"decisions": decisions, "count": len(decisions), "status": "success"}
        
        else:
            return {"error": f"Unknown operation: {operation}", "status": "error"}


def log_model():
    """Log the model to MLflow for deployment."""
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="memory_agent",
            python_model=MemoryAgentEndpoint(),
            pip_requirements=[
                "soul-agent[anthropic]",
                "databricks-sdk",
                "databricks-vector-search",
                "pyspark"
            ],
            code_path=["agents/memory"]
        )
        print("Model logged successfully")


if __name__ == "__main__":
    log_model()
