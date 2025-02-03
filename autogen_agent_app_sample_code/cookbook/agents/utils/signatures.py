from mlflow.types.schema import Array, ColSpec, DataType, Map, Object, Property, Schema

# This is a custom version of the StringResponse class from Databricks Agents
# that includes the `messages` field.
# StringResponse: from mlflow.models.rag_signatures import StringResponse

STRING_RESPONSE_WITH_MESSAGES = Schema(
    [
        ColSpec(name="content", type=DataType.string),
        ColSpec(
            name="messages",
            type=Array(
                Object(
                    [
                        Property("role", DataType.string),
                        Property("content", DataType.string, False),
                        Property("name", DataType.string, False),
                        Property("refusal", DataType.string, False),
                        Property(
                            "tool_calls",
                            Array(
                                Object(
                                    [
                                        Property("id", DataType.string),
                                        Property(
                                            "function",
                                            Object(
                                                [
                                                    Property("name", DataType.string),
                                                    Property(
                                                        "arguments", DataType.string
                                                    ),
                                                ]
                                            ),
                                        ),
                                        Property("type", DataType.string),
                                    ]
                                )
                            ),
                            False,
                        ),
                        Property("tool_call_id", DataType.string, False),
                    ]
                ),
            ),
            required=False,
        ),
    ]
)
