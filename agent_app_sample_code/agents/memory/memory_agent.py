"""
Memory-enabled agent using soul.py's hybrid RAG+RLM architecture.

Wraps the HybridAgent with Delta-backed storage for use in Databricks.
"""

from typing import Dict, Any, Optional
import os

from .delta_backend import DeltaMemoryBackend


class MemoryEnabledAgent:
    """
    Agent with persistent memory across sessions.
    
    Uses soul.py's HybridAgent internally with a Delta-backed storage layer.
    Queries are automatically routed to RAG (fast) or RLM (thorough) based
    on complexity.
    """
    
    def __init__(
        self,
        backend: DeltaMemoryBackend,
        provider: str = "anthropic",
        chat_model: Optional[str] = None,
        router_model: Optional[str] = None
    ):
        """
        Initialize memory-enabled agent.
        
        Args:
            backend: DeltaMemoryBackend instance for storage
            provider: LLM provider (anthropic, openai, gemini)
            chat_model: Model for chat completions (provider default if None)
            router_model: Model for query routing (uses cheap model if None)
        """
        self.backend = backend
        self.provider = provider
        self.chat_model = chat_model
        self.router_model = router_model
        
        # Initialize soul.py agent
        self._init_agent()
    
    def _init_agent(self):
        """Initialize the underlying HybridAgent."""
        try:
            from soul.hybrid_agent import HybridAgent
            
            # Build agent config
            config = {
                "provider": self.provider,
                "memory_backend": self.backend
            }
            
            if self.chat_model:
                config["chat_model"] = self.chat_model
            if self.router_model:
                config["router_model"] = self.router_model
            
            self.agent = HybridAgent(**config)
            
        except ImportError:
            raise ImportError(
                "soul-agent package not installed. "
                "Run: pip install soul-agent[anthropic]"
            )
    
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Ask the agent a question with memory context.
        
        The agent automatically:
        1. Retrieves relevant memories
        2. Routes to RAG or RLM based on query complexity
        3. Generates a response with memory context
        4. Logs the interaction to memory
        
        Args:
            query: The question to ask
            
        Returns:
            Dictionary with:
                - answer: The agent's response
                - route: Which retrieval strategy was used (RAG or RLM)
                - memories_used: Number of memories retrieved
        """
        result = self.agent.ask(query)
        
        # Log the interaction
        self.backend.log_memory(query, role="user")
        self.backend.log_memory(result["answer"], role="assistant")
        
        return result
    
    def remember(self, content: str, role: str = "system") -> None:
        """
        Explicitly store something in memory.
        
        Args:
            content: Content to remember
            role: Role for the memory entry (default: system)
        """
        self.backend.log_memory(content, role=role)
    
    def recall(self, query: str, k: int = 5) -> list:
        """
        Retrieve memories relevant to a query.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of relevant memory entries
        """
        return self.backend.search(query, k=k)
    
    def decide(self, decision: str, context: str = None, 
               rationale: str = None) -> None:
        """
        Log an explicit decision with rationale.
        
        Decisions are stored separately from conversation memory
        and given higher weight during retrieval.
        
        Args:
            decision: The decision that was made
            context: Context for the decision
            rationale: Reasoning behind the decision
        """
        self.backend.log_decision(decision, context, rationale)
    
    def get_identity(self) -> str:
        """Get the agent's current identity (SOUL.md content)."""
        return self.backend.get_soul()
    
    def set_identity(self, content: str) -> None:
        """
        Set or update the agent's identity.
        
        Args:
            content: New identity content (markdown)
        """
        self.backend.set_soul(content)
    
    def get_recent_memory(self, limit: int = 50) -> str:
        """
        Get recent memory as formatted markdown.
        
        Args:
            limit: Maximum entries to retrieve
            
        Returns:
            Formatted memory log
        """
        return self.backend.get_memory(limit=limit)
    
    def get_decisions(self, limit: int = 20) -> list:
        """
        Get recent decisions.
        
        Args:
            limit: Maximum decisions to retrieve
            
        Returns:
            List of decision dictionaries
        """
        return self.backend.get_decisions(limit=limit)
