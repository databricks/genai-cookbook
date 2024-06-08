## Curate an Evaluation Set from stakeholder feedback

```{image} ../images/5-hands-on/11_img.png
:align: center
```

<br/>

**Expected time:** 10 - 60 minutes

*Time varies based on the quality of the responses provided by your stakeholders.  If the responses are messy or contain lots of irrelevant queries, you will need to spend more time filtering and cleaning the data.*

**Overview & expected outcome**

This step will bootstrap an evaluation set with the feedback that stakeholders have provided by using the Review App.  Note that you can bootstrap an evaluation set with *just* questions, so even if your stakeholders only chatted with the app vs. providing feedback, you can follow this step.

Visit [documentation](https://docs.databricks.com/generative-ai/agent-evaluation/evaluation-set.html#evaluation-set-schema) to understand the Quality Lab Evaluation Set schema - these fields are referenced below.

At the end of this step, you will have an Evaluation Set that contains:

1. Requests with a 👍 :
   - `request`: As entered by the user
   - `expected_response`: If the user edited the response, that is used, otherwise, the model's generated response.
2. Requests with a 👎 :
   - `request`: As entered by the user
   - `expected_response`: If the user edited the response, that is used, otherwise, null.
3. Requests without any feedback e.g., no 👍 or 👎
   - `request`: As entered by the user

Across all of the above, if the user 👍 a chunk from the `retrieved_context`, the `doc_uri` of that chunk is included in `expected_retrieved_context` for the question.

```{important}
Databricks recommends that your Evaluation Set contain at least 30 questions to get started.  Read the [evaluation set deep dive](./4-evaluation-eval-sets.md) to learn more about what a "good" evaluation set is.
```

**Requirements:**

- Stakeholders have used your POC and provided feedback
- All requirements from previous steps

**Instructions**

1. Open the `04_create_evaluation_set` Notebook and press Run All.

2. Inspect the Evaluation Set to understand the data that is included. You need to validate that your Evaluation Set contains a representative and challenging set of questions. Adjust the Evaluation Set as required.

3. By default, your evaluation set is saved to the Delta Table configured in `EVALUATION_SET_FQN` in the `00_global_config` Notebook.

> **Next step:** Now that you have an evaluation set, use it to [evaluate the POC app's](./5-hands-on-evaluate-poc.md) quality/cost/latency.
