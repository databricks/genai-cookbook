from pydantic import BaseModel
from typing import Any, List, Literal, Dict
from utils.agents.llm import LLMConfig, LLMParametersConfig
from utils.agents.tools import (
    SerializableModel,
    load_obj_from_yaml,
    _CLASS_PATH_KEY,
    obj_to_yaml,
)
import yaml


# Design for multi-agent

# requirements
# * can test locally with just the agent's pyfunc classes
# * when you change any config, it all just reloads

# when you deploy:
# * you  deploy each supervised agent separately to model serving
# * then mutli agent picks these up
# * then mutli agent deploys

# * each child agent has [name, description, config, code]
#  - when deployed, it reads it from the UC
#  - locally, from the config

# Internal implementation details for strings that the LLM sees and may need tuning
# These constants can be adjusted to improve the quality and reliability of the LLM's responses
FINISH_ROUTE_NAME = "FINISH"  # reserved name for the finish agent which is hardcoded logic to return the last worker's response to the user
SUPERVISOR_ROUTE_NAME = "SUPERVISOR"  # reserved name for the supervisor agent which is the main agent that controls the conversation
ROUTING_FUNCTION_NAME = "decide_next_worker_or_finish"  # function name presented to the supervisor LLM via OpenAI function calling.  Used by supervisor to return it's routing decision.
WORKER_PROMPT_TEMPLATE = "<worker><worker_name>{worker_name}</worker_name><worker_description>{worker_description}</worker_description>"
# Variable names in the ROUTING_FUNCTION_NAME for the supervisor agent's outputted thinking process and decision making
CONVERSATION_HISTORY_THINKING_PARAM = "conversation_history_thinking"
WORKER_CAPABILITIES_THINKING_PARAM = "worker_capabilities_thinking"
NEXT_WORKER_OR_FINISH_PARAM = "next_worker_or_finish"


class MultiAgentSupervisorConfig(SerializableModel):
    """
    Configuration for the multi-agent supervisor.

    Attributes:
        llm_endpoint_name (str): Databricks Model Serving endpoint name for the supervisor's LLM.
        llm_parameters (LLMParametersConfig): Parameters controlling LLM response behavior.
        input_example (Any): Example input used by MLflow to set the model's input schema.
        playground_debug_mode (bool): When True, outputs debug info to playground UI. Defaults to False.
        agent_loading_mode (str): Mode for loading supervised agents - "local" or "model_serving".
        max_workers_called (int): Maximum number of worker agent turns before finishing.
        supervisor_system_prompt (str): System prompt template for the supervisor agent.
    """

    llm_endpoint_name: str
    """
    Databricks Model Serving endpoint name.
    This is the LLM used by the supervisor to make decisions.
    Databricks foundational model endpoints can be found here: https://docs.databricks.com/en/machine-learning/foundation-models/index.html
    """

    llm_parameters: LLMParametersConfig
    """
    Parameters that control how the LLM responds, including temperature and max_tokens.
    See LLMParametersConfig for details on available parameters.
    """
    input_example: Any = {
        "messages": [
            {
                "role": "user",
                "content": "What can you help me with?",
            },
        ]
    }
    """
    Example input used by MLflow to set the Agent's input schema when calling mlflow.pyfunc.log_model().
    This should match the format of inputs that will be passed to the model's predict() method.
    For chat agents, this is typically a dictionary containing a 'messages' key with an array of message objects.
    Example: {'messages': [{'role': 'user', 'content': 'Hello'}]}
    """

    playground_debug_mode: bool = False
    """
    Outputs details of all supervised agent's tool calling to the playground UI by adding it to the agent's response.
    Turn off if you don't want end users to see this debugging information, but highly recommended to keep enabled
    during development and pre-prod to visualize the agent's logic in playground/review app.
    """

    agent_loading_mode: Literal["local", "model_serving"] = "local"
    """
    Mode for loading supervised agents:
    - local: Supervised agent's code and config are loaded from your local environment. Use this mode during development for faster inner loop testing.
    - model_serving: Supervised agent is deployed as a Databricks Model Serving endpoint that gets called. Use this mode when deploying the agent to pre-prod/prod environments.
    """

    max_workers_called: int = 5
    """
    The maximum turns of conversations with the workers before the last worker's response is returned to the user by the supervisor's hard coded logic.
    """

    supervisor_system_prompt: str = """## Role
You are a supervisor responsible for managing a conversation between a user and the following workers.  You select the next worker to respond or end the conversation to return the last worker's response to the user.  Use the {ROUTING_FUNCTION_NAME} function to share your step-by-step reasoning and decision.

## Workers
<workers>{workers_names_and_descriptions}</workers>

## Objective
Your goal is to facilitate the conversation and ensure the user receives a helpful response.

## Instructions
1. **Review the Conversation History**: Think step by step by to understand the user's request and the conversation history which includes previous worker's responses.  Output to the `{CONVERSATION_HISTORY_THINKING_PARAM}` variable.
2. **Assess Worker Descriptions**: Think step by step to consider the description of each worker to understand their capabilities in the context of the conversation history.  Output to the `{WORKER_CAPABILITIES_THINKING_PARAM}` variable.
3. **Select the next worker OR finish the conversation**: Based on the converastion history, the worker's descriptions and your thinking, decide which worker should respond next OR if the conversation should finish with the last worker's response going to the user.  If the conversation becomes unproductive or stuck, also respond with "{FINISH_ROUTE_NAME}."  Output either the <worker_name> or "{FINISH_ROUTE_NAME}" to the `{NEXT_WORKER_OR_FINISH_PARAM}` variable.

## Additional Notes
- A conversation is considered "stuck" if there is no progress or if workers are unable to proceed with their tasks."""
    """
    System prompt sent to the supervisor agent before the conversation history to guide its decision-making process.
    The variable names like {ROUTING_FUNCTION_NAME}, {workers_names_and_descriptions}, etc. will be used by format() in the agent's code to populate the prompt at runtime, so do not change them.
    Improving quality: You will tune this prompt to improve the supervisor's ability to route the conversation - start with worker descriptions & names, then tune the rest of the prompt.
    """

    supervisor_user_prompt: str = (
        """Given the converastion history, the worker's descriptions and your thinking, which worker should act next OR should we FINISH? Respond with one of [{worker_names_with_finish}] to the `{NEXT_WORKER_OR_FINISH_PARAM}` variable in the {ROUTING_FUNCTION_NAME} function."""
    )
    """
    Prompt sent to supervisor after system prompt and conversation history to request next worker selection.
    The variable names will be populated at runtime via format().
    """

    agents: List[Any]
    """
    List of supervised agents that will be called by the supervisor agent. Each agent must be a  agent that implements the cookbook's Agent configuration interface.
    """

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableModel":
        # Deserialize tools, dynamically reconstructing each tool
        agents = []
        for agent_dict in data["agents"]:
            agent_yml = yaml.dump(agent_dict)
            agents.append(load_obj_from_yaml(agent_yml))

        # Replace tools with deserialized instances
        data["agents"] = agents
        return class_object(**data)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude name and description fields.

        Returns:
            Dict[str, Any]: Dictionary representation of the model excluding name and description.
        """
        model_dumped = super().model_dump(**kwargs)
        model_dumped["agents"] = [
            yaml.safe_load(obj_to_yaml(agent)) for agent in self.agents
        ]
        return model_dumped


class SupervisedAgentConfig(SerializableModel):
    name: str
    description: str
    endpoint_name: str
