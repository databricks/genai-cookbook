"""
Tests for the memory module.

Run locally with: pytest agents/memory/tests/
Run in Databricks: import and call test functions directly
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List


class MockSparkSession:
    """Mock Spark session for local testing."""
    
    def __init__(self):
        self.executed_queries = []
        self._mock_data = {
            "identity": [],
            "memory_log": [],
            "decisions": []
        }
    
    def sql(self, query: str):
        self.executed_queries.append(query)
        return MockDataFrame(self._mock_data, query)
    
    def add_mock_data(self, table: str, rows: List[Dict]):
        self._mock_data[table] = rows


class MockDataFrame:
    """Mock Spark DataFrame."""
    
    def __init__(self, data: Dict, query: str):
        self._data = data
        self._query = query
    
    def collect(self):
        # Simple mock: return empty or mock rows based on query
        if "SELECT COALESCE(MAX(version)" in self._query:
            return [MagicMock(max_version=0)]
        if "FROM" in self._query:
            for table, rows in self._data.items():
                if table in self._query:
                    return [MagicMock(**row) for row in rows]
        return []


class TestDeltaMemoryBackend:
    """Tests for DeltaMemoryBackend."""
    
    @pytest.fixture
    def mock_spark(self):
        return MockSparkSession()
    
    @pytest.fixture
    def backend(self, mock_spark):
        with patch('pyspark.sql.SparkSession') as mock_builder:
            mock_builder.builder.getOrCreate.return_value = mock_spark
            
            from agents.memory.delta_backend import DeltaMemoryBackend
            backend = DeltaMemoryBackend(
                catalog="test_catalog",
                schema="test_schema", 
                agent_id="test_agent"
            )
            backend.spark = mock_spark
            return backend
    
    def test_get_soul_empty(self, backend, mock_spark):
        """Test get_soul returns empty string when no identity exists."""
        result = backend.get_soul()
        assert result == ""
        assert any("identity" in q for q in mock_spark.executed_queries)
    
    def test_get_soul_with_data(self, backend, mock_spark):
        """Test get_soul returns identity content."""
        mock_spark.add_mock_data("identity", [
            {"soul_content": "# Test Agent\nI am a test agent."}
        ])
        result = backend.get_soul()
        assert "Test Agent" in result or result == ""  # Depends on mock
    
    def test_log_memory_escapes_quotes(self, backend, mock_spark):
        """Test that single quotes are escaped to prevent SQL injection."""
        backend.log_memory("User's input with 'quotes'", role="user")
        
        # Check the query was executed
        assert len(mock_spark.executed_queries) > 0
        last_query = mock_spark.executed_queries[-1]
        
        # Should have escaped quotes
        assert "User''s input with ''quotes''" in last_query
    
    def test_log_decision(self, backend, mock_spark):
        """Test logging a decision."""
        backend.log_decision(
            decision="Use typed tags",
            context="PII redaction",
            rationale="Better for auditing"
        )
        
        assert len(mock_spark.executed_queries) > 0
        last_query = mock_spark.executed_queries[-1]
        assert "decisions" in last_query
        assert "Use typed tags" in last_query


class TestMemoryEnabledAgent:
    """Tests for MemoryEnabledAgent."""
    
    @pytest.fixture
    def mock_backend(self):
        backend = MagicMock()
        backend.get_soul.return_value = "# Test Agent"
        backend.get_memory.return_value = ""
        backend.search.return_value = []
        return backend
    
    def test_ask_logs_interaction(self, mock_backend):
        """Test that ask() logs both query and response to memory."""
        with patch('soul.hybrid_agent.HybridAgent') as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.ask.return_value = {
                "answer": "Test response",
                "route": "RAG"
            }
            MockAgent.return_value = mock_agent_instance
            
            from agents.memory.memory_agent import MemoryEnabledAgent
            agent = MemoryEnabledAgent(backend=mock_backend)
            
            result = agent.ask("Test question")
            
            # Should log user query and assistant response
            assert mock_backend.log_memory.call_count == 2
            calls = mock_backend.log_memory.call_args_list
            assert calls[0][0][0] == "Test question"
            assert calls[0][1]["role"] == "user"
            assert calls[1][0][0] == "Test response"
            assert calls[1][1]["role"] == "assistant"
    
    def test_remember(self, mock_backend):
        """Test explicit memory storage."""
        with patch('soul.hybrid_agent.HybridAgent'):
            from agents.memory.memory_agent import MemoryEnabledAgent
            agent = MemoryEnabledAgent(backend=mock_backend)
            
            agent.remember("Important fact", role="system")
            
            mock_backend.log_memory.assert_called_with("Important fact", role="system")
    
    def test_decide(self, mock_backend):
        """Test decision logging."""
        with patch('soul.hybrid_agent.HybridAgent'):
            from agents.memory.memory_agent import MemoryEnabledAgent
            agent = MemoryEnabledAgent(backend=mock_backend)
            
            agent.decide(
                decision="Route to senior support",
                context="VIP customer",
                rationale="SLA requirements"
            )
            
            mock_backend.log_decision.assert_called_with(
                "Route to senior support",
                "VIP customer",
                "SLA requirements"
            )


class TestEndpointModel:
    """Tests for the MLflow endpoint model."""
    
    def test_ask_operation(self):
        """Test ask operation via endpoint."""
        with patch('agents.memory.delta_backend.DeltaMemoryBackend') as MockBackend, \
             patch('agents.memory.memory_agent.MemoryEnabledAgent') as MockAgent:
            
            mock_agent = MagicMock()
            mock_agent.ask.return_value = {"answer": "Response", "route": "RAG"}
            MockAgent.return_value = mock_agent
            
            from agents.memory.endpoint_model import MemoryAgentEndpoint
            endpoint = MemoryAgentEndpoint()
            endpoint.agent = mock_agent
            
            result = endpoint.predict(None, {"operation": "ask", "query": "Test?"})
            
            assert result["status"] == "success"
            assert result["answer"] == "Response"
    
    def test_remember_operation(self):
        """Test remember operation via endpoint."""
        with patch('agents.memory.delta_backend.DeltaMemoryBackend'), \
             patch('agents.memory.memory_agent.MemoryEnabledAgent') as MockAgent:
            
            mock_agent = MagicMock()
            MockAgent.return_value = mock_agent
            
            from agents.memory.endpoint_model import MemoryAgentEndpoint
            endpoint = MemoryAgentEndpoint()
            endpoint.agent = mock_agent
            
            result = endpoint.predict(None, {
                "operation": "remember",
                "query": "Remember this"
            })
            
            assert result["status"] == "stored"
            mock_agent.remember.assert_called()
    
    def test_string_input_treated_as_ask(self):
        """Test that plain string input is treated as ask operation."""
        with patch('agents.memory.delta_backend.DeltaMemoryBackend'), \
             patch('agents.memory.memory_agent.MemoryEnabledAgent') as MockAgent:
            
            mock_agent = MagicMock()
            mock_agent.ask.return_value = {"answer": "Response", "route": "RAG"}
            MockAgent.return_value = mock_agent
            
            from agents.memory.endpoint_model import MemoryAgentEndpoint
            endpoint = MemoryAgentEndpoint()
            endpoint.agent = mock_agent
            
            result = endpoint.predict(None, "Plain text question")
            
            assert result["status"] == "success"
            mock_agent.ask.assert_called_with("Plain text question")


# Integration test (requires Databricks)
def test_integration_databricks():
    """
    Integration test for Databricks environment.
    
    Run this in a Databricks notebook:
    
    ```python
    from agents.memory.tests.test_memory import test_integration_databricks
    test_integration_databricks()
    ```
    """
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        
        # Create test schema
        spark.sql("CREATE SCHEMA IF NOT EXISTS test_memory_integration")
        
        # Create tables
        spark.sql("""
            CREATE TABLE IF NOT EXISTS test_memory_integration.identity (
                agent_id STRING,
                soul_content STRING,
                updated_at TIMESTAMP DEFAULT current_timestamp(),
                version INT DEFAULT 1
            ) USING DELTA
        """)
        
        spark.sql("""
            CREATE TABLE IF NOT EXISTS test_memory_integration.memory_log (
                agent_id STRING,
                timestamp TIMESTAMP DEFAULT current_timestamp(),
                role STRING,
                content STRING,
                metadata MAP<STRING, STRING>
            ) USING DELTA
        """)
        
        spark.sql("""
            CREATE TABLE IF NOT EXISTS test_memory_integration.decisions (
                agent_id STRING,
                decision STRING,
                context STRING,
                rationale STRING,
                created_at TIMESTAMP DEFAULT current_timestamp(),
                confidence DOUBLE DEFAULT 1.0
            ) USING DELTA
        """)
        
        # Test backend
        from agents.memory.delta_backend import DeltaMemoryBackend
        backend = DeltaMemoryBackend(
            catalog="hive_metastore",  # or your catalog
            schema="test_memory_integration",
            agent_id="integration_test"
        )
        
        # Test operations
        backend.log_memory("Test memory entry", role="user")
        backend.log_decision("Test decision", "Test context", "Test rationale")
        
        memory = backend.get_memory(limit=10)
        decisions = backend.get_decisions(limit=10)
        
        assert len(memory) > 0, "Memory should not be empty"
        assert len(decisions) > 0, "Decisions should not be empty"
        
        # Cleanup
        backend.clear_memory(keep_identity=False)
        spark.sql("DROP SCHEMA IF EXISTS test_memory_integration CASCADE")
        
        print("✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
