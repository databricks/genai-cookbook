from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from typing import List, Optional


class DatabricksModelServingClient:
    def __init__(self, config, **kwargs):
        self.workspace = WorkspaceClient()
        self.openai_client = self.workspace.serving_endpoints.get_open_ai_client()
        self.endpoint_name = config.get("endpoint_name")
        self.llm_config = config.get("llm_config")

    def create(self, input_data):
      messages = []
      for message in input_data['messages']:
        message.pop("name", None)
        messages.append(message)

      llm_config = self.llm_config.copy()

      if 'tools' in input_data:
        llm_config["tools"] = input_data["tools"]
        llm_config["tool_choice"] = "auto"

      response = self.openai_client.chat.completions.create(
          model=self.endpoint_name,
          messages=messages,
          **self.llm_config
      )
      
      return response

    def message_retrieval(self, response):
      # Process and return messages from the response
      return [choice.message for choice in response.choices]

    def cost(self, response):
      # Implement cost calculation if applicable
      return 0

    def get_usage(self, response):
      usage = response.usage
      # Implement usage statistics if available
      return {"prompt_tokens": usage.prompt_tokens, "total_tokens": usage.total_tokens, "completion_tokens": usage.completion_tokens}