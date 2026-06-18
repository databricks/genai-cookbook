"""
Delta-backed storage for soul.py memory.

Stores agent identity, memory log, and decisions in Unity Catalog Delta tables.
"""

from pyspark.sql import SparkSession
from datetime import datetime
from typing import List, Dict, Any, Optional


class DeltaMemoryBackend:
    """Store soul.py memory in Delta tables for distributed, governed access."""
    
    def __init__(self, catalog: str, schema: str, agent_id: str):
        """
        Initialize Delta memory backend.
        
        Args:
            catalog: Unity Catalog catalog name
            schema: Schema containing memory tables
            agent_id: Unique identifier for this agent instance
        """
        self.catalog = catalog
        self.schema = schema
        self.agent_id = agent_id
        self.spark = SparkSession.builder.getOrCreate()
        self._table_prefix = f"{catalog}.{schema}"
    
    def get_soul(self) -> str:
        """
        Retrieve agent identity (SOUL.md equivalent).
        
        Returns:
            Agent identity content, or empty string if not set
        """
        df = self.spark.sql(f"""
            SELECT soul_content 
            FROM {self._table_prefix}.identity
            WHERE agent_id = '{self.agent_id}'
            ORDER BY version DESC
            LIMIT 1
        """)
        rows = df.collect()
        return rows[0].soul_content if rows else ""
    
    def set_soul(self, content: str) -> None:
        """
        Set or update agent identity.
        
        Args:
            content: New identity content (markdown)
        """
        # Get current version
        current = self.spark.sql(f"""
            SELECT COALESCE(MAX(version), 0) as max_version
            FROM {self._table_prefix}.identity
            WHERE agent_id = '{self.agent_id}'
        """).collect()[0].max_version
        
        # Insert new version
        escaped_content = content.replace("'", "''")
        self.spark.sql(f"""
            INSERT INTO {self._table_prefix}.identity
            (agent_id, soul_content, version)
            VALUES ('{self.agent_id}', '{escaped_content}', {current + 1})
        """)
    
    def get_memory(self, limit: int = 100) -> str:
        """
        Retrieve recent memory (MEMORY.md equivalent).
        
        Args:
            limit: Maximum number of entries to retrieve
            
        Returns:
            Formatted markdown of recent memory entries
        """
        df = self.spark.sql(f"""
            SELECT timestamp, role, content
            FROM {self._table_prefix}.memory_log
            WHERE agent_id = '{self.agent_id}'
            ORDER BY timestamp DESC
            LIMIT {limit}
        """)
        
        # Format as markdown
        lines = []
        for row in reversed(df.collect()):
            ts = row.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"[{ts}] **{row.role}**: {row.content}")
        
        return "\n".join(lines)
    
    def log_memory(self, content: str, role: str = "user", 
                   metadata: Optional[Dict[str, str]] = None) -> None:
        """
        Append an entry to the memory log.
        
        Args:
            content: Memory content to store
            role: Role (user, assistant, system)
            metadata: Optional key-value metadata
        """
        escaped_content = content.replace("'", "''")
        metadata_sql = "MAP()" if not metadata else f"MAP({', '.join(f\"'{k}', '{v}'\" for k, v in metadata.items())})"
        
        self.spark.sql(f"""
            INSERT INTO {self._table_prefix}.memory_log
            (agent_id, role, content, metadata)
            VALUES ('{self.agent_id}', '{role}', '{escaped_content}', {metadata_sql})
        """)
    
    def log_decision(self, decision: str, context: str = None, 
                     rationale: str = None, confidence: float = 1.0) -> None:
        """
        Record an explicit decision with rationale.
        
        Args:
            decision: The decision that was made
            context: Context in which the decision was made
            rationale: Reasoning behind the decision
            confidence: Confidence level (0-1)
        """
        decision_escaped = decision.replace("'", "''")
        context_escaped = (context or "").replace("'", "''")
        rationale_escaped = (rationale or "").replace("'", "''")
        
        self.spark.sql(f"""
            INSERT INTO {self._table_prefix}.decisions
            (agent_id, decision, context, rationale, confidence)
            VALUES (
                '{self.agent_id}', 
                '{decision_escaped}', 
                '{context_escaped}', 
                '{rationale_escaped}',
                {confidence}
            )
        """)
    
    def get_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve recent decisions.
        
        Args:
            limit: Maximum number of decisions to retrieve
            
        Returns:
            List of decision dictionaries
        """
        df = self.spark.sql(f"""
            SELECT decision, context, rationale, confidence, created_at
            FROM {self._table_prefix}.decisions
            WHERE agent_id = '{self.agent_id}'
            ORDER BY created_at DESC
            LIMIT {limit}
        """)
        
        return [row.asDict() for row in df.collect()]
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over memories using Vector Search.
        
        Requires a Vector Search index on the memory_log table.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of matching memory entries
        """
        try:
            from databricks.vector_search.client import VectorSearchClient
            vs = VectorSearchClient()
            
            index_name = f"{self._table_prefix}.memory_index"
            results = vs.index(index_name).similarity_search(
                query_text=query,
                columns=["timestamp", "role", "content"],
                num_results=k,
                filters={"agent_id": self.agent_id}
            )
            return results.get("result", {}).get("data_array", [])
        except Exception as e:
            # Fall back to keyword search if Vector Search not available
            escaped_query = query.replace("'", "''")
            df = self.spark.sql(f"""
                SELECT timestamp, role, content
                FROM {self._table_prefix}.memory_log
                WHERE agent_id = '{self.agent_id}'
                  AND LOWER(content) LIKE '%{escaped_query.lower()}%'
                ORDER BY timestamp DESC
                LIMIT {k}
            """)
            return [row.asDict() for row in df.collect()]
    
    def clear_memory(self, keep_identity: bool = True) -> None:
        """
        Clear agent memory (useful for testing).
        
        Args:
            keep_identity: If True, preserve identity; if False, clear everything
        """
        self.spark.sql(f"""
            DELETE FROM {self._table_prefix}.memory_log
            WHERE agent_id = '{self.agent_id}'
        """)
        self.spark.sql(f"""
            DELETE FROM {self._table_prefix}.decisions
            WHERE agent_id = '{self.agent_id}'
        """)
        
        if not keep_identity:
            self.spark.sql(f"""
                DELETE FROM {self._table_prefix}.identity
                WHERE agent_id = '{self.agent_id}'
            """)
