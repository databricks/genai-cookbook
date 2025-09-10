import mlflow
from typing import List, Dict
import json

##
# Utility functions for formatting OpenAI tool calls and responses for display in Databricks
# playground and review applications. These functions convert the raw message format into
# a more readable, XML-tagged format suitable for UI rendering.
##


@mlflow.trace(span_type="PARSER")
def convert_messages_to_playground_tool_display_strings(
    messages: List[Dict[str, str]]
) -> str:
    """Format a list of OpenAI chat messages for display in Databricks playground/review UI.

    Processes a sequence of OpenAI chat messages, with special handling for tool calls
    and their responses. Tool-related content is wrapped in XML-like tags for proper
    UI rendering and readability.

    Args:
        messages (List[Dict[str, str]]): List of OpenAI message dictionaries containing role
            (user/assistant/tool), content, and optional tool_calls from the chat completion API.

    Returns:
        str: UI-friendly string with tool calls wrapped in <uc_function_call> tags and
            tool responses wrapped in <uc_function_result> tags.
    """
    output = ""
    for msg in messages:  # ignore first user input
        if msg["role"] == "assistant" and msg.get("tool_calls"):  # tool call
            for tool_call in msg["tool_calls"]:
                output += stringify_tool_call(tool_call)
            # output += f"<uc_function_call>{json.dumps(msg, indent=2)}</uc_function_call>"
        elif msg["role"] == "tool":  # tool response
            output += stringify_tool_result(msg)
            # output += f"<uc_function_result>{json.dumps(msg, indent=2)}</uc_function_result>"
        else:
            output += msg["content"] if msg["content"] != None else ""
    return output


@mlflow.trace(span_type="PARSER")
def stringify_tool_call(tool_call) -> str:
    """Format an OpenAI tool call for display in Databricks playground/review UI.

    Extracts relevant information from an OpenAI tool call and formats it into a
    UI-friendly string wrapped in XML-like tags for proper rendering.

    Args:
        tool_call (dict): OpenAI tool call dictionary containing function details
            (name, arguments) and call ID from the chat completion API.

    Returns:
        str: UI-friendly string wrapped in <uc_function_call> tags, containing the
            tool's name, ID, and arguments in a structured format.
    """
    try:
        function = tool_call["function"]
        args_dict = json.loads(function["arguments"])
        request = {
            "id": tool_call["id"],
            "name": function["name"],
            "arguments": json.dumps(args_dict),
        }

        return f"<uc_function_call>{json.dumps(request)}</uc_function_call>"

    except Exception as e:
        print("Failed to stringify tool call: ", e)
        return str(tool_call)


@mlflow.trace(span_type="PARSER")
def stringify_tool_result(tool_msg) -> str:
    """Format an OpenAI tool response for display in Databricks playground/review UI.

    Processes a tool's response message and formats it into a UI-friendly string
    wrapped in XML-like tags for proper rendering.

    Args:
        tool_msg (dict): OpenAI tool response dictionary containing the tool_call_id
            and response content from the chat completion API.

    Returns:
        str: UI-friendly string wrapped in <uc_function_result> tags, containing the
            tool's response ID and content.
    """
    try:

        result = json.dumps(
            {"id": tool_msg["tool_call_id"], "content": tool_msg["content"]}
        )
        return f"<uc_function_result>{result}</uc_function_result>"
    except Exception as e:
        print("Failed to stringify tool result:", e)
        return str(tool_msg)
