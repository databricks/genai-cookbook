from pydantic import BaseModel

class LLMParametersConfig(BaseModel):
    """
    Configuration for LLM response parameters.

    Attributes:
        temperature (float): Controls randomness in the response.
        max_tokens (int): Maximum number of tokens in the response.
        top_p (float): Controls diversity via nucleus sampling.
        top_k (int): Limits the number of highest probability tokens considered.
    """

    # Parameters that control how the LLM responds.
    temperature: float = None
    max_tokens: int = None


class LLMConfig(BaseModel):
    """
    Configuration for the function-calling LLM.

    Attributes:
        llm_endpoint_name (str): Databricks Model Serving endpoint name.
            This is the generator LLM where your LLM queries are sent.
            Databricks foundational model endpoints can be found here:
            https://docs.databricks.com/en/machine-learning/foundation-models/index.html
        llm_system_prompt_template (str): Template for the LLM prompt.
            This is how the RAG chain combines the user's question and the retrieved context.
        llm_parameters (LLMParametersConfig): Parameters that control how the LLM responds.
    """

    # Databricks Model Serving endpoint name
    # This is the generator LLM where your LLM queries are sent.
    # Databricks foundational model endpoints can be found here: https://docs.databricks.com/en/machine-learning/foundation-models/index.html
    llm_endpoint_name: str

    # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
    llm_system_prompt_template: str

    # Parameters that control how the LLM responds.
    llm_parameters: LLMParametersConfig
