## Establishing Ground Truth: Creating Evaluation Sets

To measure quality, Databricks recommends creating a human-labeled Evaluation Set, which is a curated, representative set of queries, along with ground-truth answers and (optionally) the correct supporting documents that should be retrieved. Human input is crucial in this process, as it ensures that the Evaluation Set accurately reflects the expectations and requirements of the end-users.

A good Evaluation Set has the following characteristics:

- **Representative:** Accurately reflects the variety of requests the application will encounter in production.
- **Challenging:** The set should include difficult and diverse cases to effectively test the model's capabilities.  Ideally, it will include adversarial examples such as questions attempting prompt injection or questions attempting to generate inappropriate responses from LLM.
- **Continually updated:** The set must be periodically updated to reflect how the application is used in production and the changing nature of the indexed data.

Databricks recommends at least 30 questions in your evaluation set, and ideally 100 - 200. The best evaluation sets will grow over time to contain 1,000s of questions.

To avoid overfitting, Databricks recommends splitting your evaluation set into training, test, and validation sets:

- Training set: ~70% of the questions. Used for an initial pass to evaluate every experiment to identify the highest potential ones.
- Test set: ~20% of the questions. Used for evaluating the highest performing experiments from the training set.  
- Validation set: ~10% of the questions. Used for a final validation check before deploying an experiment to production.