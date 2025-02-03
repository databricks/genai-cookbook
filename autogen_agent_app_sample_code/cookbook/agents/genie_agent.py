from typing import Dict, List, Optional, Union
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from databricks_langchain.genie import Genie
import mlflow

class GenieAgent(ConversableAgent):
  def __init__(self, name, genie_space_id, **kwargs):
    super().__init__(name, llm_config=False, **kwargs)
    self.genie_api = Genie(genie_space_id)
    self.register_reply([Agent, None], GenieAgent.generate_genie_reply)
  
  @mlflow.trace()
  def _concat_messages_array(self, messages):
      concatenated_message = "\n".join([message['content'] for message in messages if message['content']])
      return concatenated_message
  
  @mlflow.trace()
  def generate_genie_reply(self, messages: Optional[List[Dict]], sender: "Agent", config):

    if messages is None:
      messages = self._oai_messages[sender]

    message = f"I will provide you a chat history, where your name is {self.name}. Please help with the described information in the chat history.\n"

    message += self._concat_messages_array(messages)
    
    genie_response = self.genie_api.ask_question(message)

    return True, {"content": genie_response.result}