## Defining "quality": evaluation sets

To measure quality, Databricks recommends creating a human-labeled evaluation set, which is a curated, representative set of queries, along with ground-truth answers and (optionally) the correct supporting documents that should be retrieved. Human input is crucial in this process, as it ensures that the evaluation set accurately reflects the expectations and requirements of the end-users.

> Curating human labels can be a time-consuming process.  You can get started by creating an evaluation set that *only* includes questions, and add the ground truth responses over time.  [Mosaic AI Quality Lab](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) can assess your chain's quality without ground truth, although, if ground truth is available, it will compute additional metrics such as answer correctness.

A good evaluation set has the following characteristics:

- **Representative:** Accurately reflects the variety of requests the application will encounter in production.
- **Challenging:** The set should include difficult and diverse cases to effectively test the model's capabilities.  Ideally, it will include adversarial examples such as questions attempting prompt injection or questions attempting to generate inappropriate responses from LLM.
- **Continually updated:** The set must be periodically updated to reflect how the application is used in production, the changing nature of the indexed data, and any changes to the application requirements.

Databricks recommends at least 30 questions in your evaluation set, and ideally 100 - 200. The best evaluation sets will grow over time to contain 1,000s of questions.

To avoid overfitting, Databricks recommends splitting your evaluation set into training, test, and validation sets:

- Training set: ~70% of the questions. Used for an initial pass to evaluate every experiment to identify the highest potential ones.
- Test set: ~20% of the questions. Used for evaluating the highest performing experiments from the training set.  
- Validation set: ~10% of the questions. Used for a final validation check before deploying an experiment to production.

> [Mosaic AI Quality Lab](https://docs.databricks.com/generative-ai/agent-evaluation/index.html) helps you create an evaluation set by providing a web-based chat interface for your stakeholders to provide feedback on the application's outputs.  The chain's outputs and the stakeholder feedback are saved in Delta Tables, which can then be curated into an evaluation set.  See [curating an evaluation set](./5-hands-on-curate-eval-set.md) in the implement section of this cookbook for a hands-on how to with sample code.