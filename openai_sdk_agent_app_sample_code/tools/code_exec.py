def python_exec(code: str) -> str:
    """
    Executes Python code in the sandboxed environment and returns its stdout. The runtime is stateless and you can not read output of the previous tool executions. i.e. No such variables "rows", "observation" defined. Calling another tool inside a Python code is NOT allowed.
    Use only standard python libraries and these python libraries: bleach, chardet, charset-normalizer, defusedxml, googleapis-common-protos, grpcio, grpcio-status, jmespath, joblib, numpy, packaging, pandas, patsy, protobuf, pyarrow, pyparsing, python-dateutil, pytz, scikit-learn, scipy, setuptools, six, threadpoolctl, webencodings, user-agents, cryptography.

    Args:
      code (str): Python code to execute. Remember to print the final result to stdout.

    Returns:
      str: The output of the executed code.
    """
    import sys
    from io import StringIO

    sys_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    exec(code)
    sys.stdout = sys_stdout
    return redirected_output.getvalue()
