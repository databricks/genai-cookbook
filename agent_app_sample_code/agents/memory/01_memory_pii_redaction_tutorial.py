# Databricks notebook source
# MAGIC %md
# MAGIC # Memory-Enabled PII Redaction with ai_query()
# MAGIC 
# MAGIC This notebook demonstrates how to add **persistent memory** to `ai_query()` calls.
# MAGIC 
# MAGIC **The problem:** Standard `ai_query()` is stateless. When the model misses a PII pattern (like phone extensions), you fix the prompt and move on. Next week, you hit another edge case — and you've forgotten the first one.
# MAGIC 
# MAGIC **The solution:** A memory layer that:
# MAGIC 1. Stores decisions when patterns are missed
# MAGIC 2. Automatically retrieves relevant decisions on future calls
# MAGIC 3. Learns from accumulated feedback over time
# MAGIC 
# MAGIC Based on [soul.py](https://github.com/menonpg/soul.py) — persistent memory for LLM agents.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup: Create Memory Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Create schema for agent memory
# MAGIC CREATE SCHEMA IF NOT EXISTS main.agent_memory;
# MAGIC 
# MAGIC -- Identity table (agent instructions/personality)
# MAGIC CREATE TABLE IF NOT EXISTS main.agent_memory.identity (
# MAGIC     agent_id STRING NOT NULL,
# MAGIC     soul_content TEXT NOT NULL,
# MAGIC     updated_at TIMESTAMP DEFAULT current_timestamp(),
# MAGIC     version INT DEFAULT 1
# MAGIC ) USING DELTA;
# MAGIC 
# MAGIC -- Memory log (conversation history)
# MAGIC CREATE TABLE IF NOT EXISTS main.agent_memory.memory_log (
# MAGIC     agent_id STRING NOT NULL,
# MAGIC     timestamp TIMESTAMP DEFAULT current_timestamp(),
# MAGIC     role STRING NOT NULL,
# MAGIC     content TEXT NOT NULL,
# MAGIC     metadata MAP<STRING, STRING>
# MAGIC ) USING DELTA
# MAGIC PARTITIONED BY (agent_id);
# MAGIC 
# MAGIC -- Decisions (explicit choices with rationale)
# MAGIC CREATE TABLE IF NOT EXISTS main.agent_memory.decisions (
# MAGIC     agent_id STRING NOT NULL,
# MAGIC     decision TEXT NOT NULL,
# MAGIC     context TEXT,
# MAGIC     rationale TEXT,
# MAGIC     created_at TIMESTAMP DEFAULT current_timestamp(),
# MAGIC     confidence DOUBLE DEFAULT 1.0
# MAGIC ) USING DELTA;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Set Up the PII Redaction Agent Identity

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Define the agent's identity (like a system prompt that persists)
# MAGIC INSERT INTO main.agent_memory.identity (agent_id, soul_content, version)
# MAGIC VALUES (
# MAGIC   'pii-redactor',
# MAGIC   '# PII Redaction Agent
# MAGIC 
# MAGIC You are a PII redaction system for healthcare data.
# MAGIC 
# MAGIC ## Replacement Tags
# MAGIC - Person names → [NAME]
# MAGIC - Social Security Numbers → [SSN]
# MAGIC - Medical Record Numbers → [MRN]
# MAGIC - Dates of birth → [DOB]
# MAGIC - Street addresses → [ADDRESS]
# MAGIC - Phone numbers → [PHONE]
# MAGIC 
# MAGIC ## Rules
# MAGIC - Preserve all clinical content: symptoms, diagnoses, medications, dosages
# MAGIC - Only redact identifiers of people
# MAGIC - Output only the redacted text, no explanation
# MAGIC 
# MAGIC ## Learned Patterns
# MAGIC Check the decisions table for patterns discovered during production use.',
# MAGIC   1
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Standard PII Redaction (Stateless)

# COMMAND ----------

# Sample clinical records
sample_records = [
    {"id": 1, "notes": "Patient John Smith (DOB 03/15/1982) called from 555-1234 x789 about fatigue."},
    {"id": 2, "notes": "Dr. Sarah Chen discussed case with family member Jane Smith at 555-867-5309."},
    {"id": 3, "notes": "Follow-up scheduled for patient MRN 487289, SSN 123-45-6789."}
]

df = spark.createDataFrame(sample_records)
df.createOrReplaceTempView("clinical_records")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Standard ai_query() PII redaction (no memory)
# MAGIC SELECT 
# MAGIC   id,
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-3-70b-instruct',
# MAGIC     concat(
# MAGIC       'Redact all PII from this clinical note. Use [NAME], [DOB], [PHONE], [MRN], [SSN] as replacement tags. ',
# MAGIC       'Return only the redacted text.\n\nNote: ', notes
# MAGIC     )
# MAGIC   ) AS redacted_notes
# MAGIC FROM clinical_records

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Discover a Missed Pattern
# MAGIC 
# MAGIC Notice that `555-1234 x789` might not be fully redacted — the extension format `x789` is unusual.
# MAGIC 
# MAGIC In production, a reviewer catches this. Now we need to **teach the agent**.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Log the decision so future calls know about this pattern
# MAGIC INSERT INTO main.agent_memory.decisions (agent_id, decision, context, rationale)
# MAGIC VALUES (
# MAGIC   'pii-redactor',
# MAGIC   'Phone numbers with extensions (format: xxx-xxxx xNNN or ext. NNN) must be fully redacted as [PHONE]',
# MAGIC   'PII redaction for clinical notes',
# MAGIC   'Discovered in production: 555-1234 x789 was only partially redacted. Extensions are part of the phone number and constitute PII.'
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Memory-Enhanced PII Redaction
# MAGIC 
# MAGIC Now we build a memory-aware prompt that includes learned decisions.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- View accumulated decisions
# MAGIC SELECT decision, rationale, created_at
# MAGIC FROM main.agent_memory.decisions
# MAGIC WHERE agent_id = 'pii-redactor'
# MAGIC ORDER BY created_at DESC

# COMMAND ----------

# Build memory-enhanced prompt
def build_memory_prompt(agent_id: str) -> str:
    """Build a prompt that includes agent identity and learned decisions."""
    
    # Get identity
    identity_df = spark.sql(f"""
        SELECT soul_content FROM main.agent_memory.identity
        WHERE agent_id = '{agent_id}'
        ORDER BY version DESC LIMIT 1
    """)
    identity = identity_df.collect()[0].soul_content if identity_df.count() > 0 else ""
    
    # Get recent decisions
    decisions_df = spark.sql(f"""
        SELECT decision, rationale FROM main.agent_memory.decisions
        WHERE agent_id = '{agent_id}'
        ORDER BY created_at DESC LIMIT 10
    """)
    
    decisions_text = ""
    if decisions_df.count() > 0:
        decisions_text = "\n\n## Learned Decisions (apply these)\n"
        for row in decisions_df.collect():
            decisions_text += f"- {row.decision}\n  Rationale: {row.rationale}\n"
    
    return identity + decisions_text

# COMMAND ----------

memory_prompt = build_memory_prompt('pii-redactor')
print(memory_prompt)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Memory-enhanced redaction (includes learned patterns)
# MAGIC SELECT 
# MAGIC   id,
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-3-70b-instruct',
# MAGIC     concat(
# MAGIC       -- Include the full memory context
# MAGIC       (SELECT concat(soul_content, '\n\n## Learned Decisions\n', 
# MAGIC                      collect_list(concat('- ', decision))[0])
# MAGIC        FROM main.agent_memory.identity i
# MAGIC        LEFT JOIN main.agent_memory.decisions d ON i.agent_id = d.agent_id
# MAGIC        WHERE i.agent_id = 'pii-redactor'
# MAGIC        GROUP BY soul_content
# MAGIC        LIMIT 1),
# MAGIC       '\n\n---\nRedact this note:\n', 
# MAGIC       notes
# MAGIC     )
# MAGIC   ) AS redacted_notes
# MAGIC FROM clinical_records

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Continuous Learning Loop
# MAGIC 
# MAGIC As reviewers catch more edge cases, add them to the decisions table:

# COMMAND ----------

# MAGIC %sql
# MAGIC -- More patterns discovered over time
# MAGIC INSERT INTO main.agent_memory.decisions (agent_id, decision, context, rationale)
# MAGIC VALUES 
# MAGIC   ('pii-redactor', 
# MAGIC    'Fax numbers should be redacted as [FAX], not [PHONE]',
# MAGIC    'Document contains "Fax: 555-1234"',
# MAGIC    'Fax numbers are also contact PII but should be labeled distinctly for audit purposes'),
# MAGIC   ('pii-redactor',
# MAGIC    'Email addresses in clinical notes should be redacted as [EMAIL]',
# MAGIC    'Note contained "contact patient at john.smith@email.com"',
# MAGIC    'Email addresses were not in original tag list but are clearly PII');

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pattern Summary
# MAGIC 
# MAGIC | Step | Action | SQL |
# MAGIC |------|--------|-----|
# MAGIC | 1 | Set agent identity | `INSERT INTO identity` |
# MAGIC | 2 | Run redaction | `ai_query()` with memory context |
# MAGIC | 3 | Catch missed pattern | Manual review |
# MAGIC | 4 | Teach the agent | `INSERT INTO decisions` |
# MAGIC | 5 | Future calls include learning | Memory prompt auto-includes decisions |
# MAGIC 
# MAGIC The agent gets **better over time** without changing the code — just by accumulating decisions.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup (Optional)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Remove test data
# MAGIC -- DELETE FROM main.agent_memory.decisions WHERE agent_id = 'pii-redactor';
# MAGIC -- DELETE FROM main.agent_memory.identity WHERE agent_id = 'pii-redactor';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. **Deploy as Model Serving endpoint** — See `endpoint_model.py` for MLflow packaging
# MAGIC 2. **Add Vector Search** — Enable semantic retrieval over memories for large decision sets
# MAGIC 3. **Integrate with Lakeflow** — Auto-extract patterns from redaction failures
# MAGIC 
# MAGIC ## References
# MAGIC 
# MAGIC - [soul.py](https://github.com/menonpg/soul.py) — Persistent memory library
# MAGIC - [ai_query() docs](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query)
# MAGIC - [LoCoMo Benchmarks](https://menonpg.github.io/soul-benchmarks/) — Memory retrieval evaluation
