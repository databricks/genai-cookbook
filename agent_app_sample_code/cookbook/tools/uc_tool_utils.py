import mlflow
from pyspark.errors import SparkRuntimeException
from pyspark.errors.exceptions.connect import SparkException


import logging
from typing import Dict, Union

ERROR_KEY = "error_message"
STACK_TRACE_KEY = "stack_trace"


@mlflow.trace(span_type="PARSER")
def _remove_udfbody_from_pyspark_stack_trace(stack_trace: str) -> str:
    return stack_trace.replace('File "<udfbody>",', "").strip()


@mlflow.trace(span_type="PARSER")
def _parse_PySpark_exception_dumped_as_string(error_msg: str) -> Dict[str, str]:
    # Extract error section between == Error == and == Stacktrace ==
    error = error_msg.split("== Error ==")[1].split("== Stacktrace ==")[0].strip()

    # Extract stacktrace section after == Stacktrace == and before SQL
    stack_trace = error_msg.split("== Stacktrace ==")[1].split("== SQL")[0].strip()

    # Remove SQLSTATE and anything after it from the stack trace
    if "SQLSTATE" in stack_trace:
        stack_trace = stack_trace.split("SQLSTATE")[0].strip()

    return {
        STACK_TRACE_KEY: _remove_udfbody_from_pyspark_stack_trace(stack_trace),
        ERROR_KEY: error,
    }


@mlflow.trace(span_type="PARSER")
def _parse_PySpark_exception_from_known_structure(
    tool_exception: Union[SparkRuntimeException, SparkException]
) -> Dict[str, str]:
    raw_stack_trace = tool_exception.getMessageParameters()["stack"]
    return {
        STACK_TRACE_KEY: _remove_udfbody_from_pyspark_stack_trace(raw_stack_trace),
        ERROR_KEY: tool_exception.getMessageParameters()["error"],
    }


@mlflow.trace(span_type="PARSER")
def _parse_generic_tool_exception(tool_exception: Exception) -> Dict[str, str]:
    return {
        STACK_TRACE_KEY: None,
        ERROR_KEY: str(tool_exception),
    }


@mlflow.trace(span_type="PARSER")
def _return_raw_PySpark_exception(
    tool_exception: Union[SparkRuntimeException, SparkException]
):
    return {
        STACK_TRACE_KEY: None,
        ERROR_KEY: str(tool_exception),
    }


@mlflow.trace(span_type="PARSER")
def _parse_SparkException_from_tool_execution(
    tool_exception: Union[SparkRuntimeException, SparkException, Exception],
) -> Dict[str, str]:
    error_info_to_return: Union[Dict, str] = None

    # First attempt: first try to parse from the known structure
    try:
        logging.info(
            f"Trying to parse spark exception {tool_exception} using its provided structured data."
        )
        # remove the <udfbody> from the stack trace which the LLM knows nothing about
        # raw_stack_trace = tool_exception.getMessageParameters()["stack"]
        return _parse_PySpark_exception_from_known_structure(tool_exception)

    except Exception as e:
        # 2nd attempt: that failed, let's try to parse the SparkException's raw formatting
        logging.error(f"Error parsing: {e}, trying alternative approaches to parsing.")

        logging.info(
            f"Trying to parse spark exception {tool_exception} using its raw string output."
        )
        try:
            raw_error_msg = str(tool_exception)
            return _parse_PySpark_exception_dumped_as_string(raw_error_msg)
        except Exception as e:
            # Last attempt: if that fails, just use the raw error
            logging.error(
                f"Error parsing spark exception using its raw string formatting: {e}, will return the raw error message."
            )

            logging.info(f"returning the raw error message: {str(tool_exception)}.")
            return _return_raw_PySpark_exception(tool_exception)
