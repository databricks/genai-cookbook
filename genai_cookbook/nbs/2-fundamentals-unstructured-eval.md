## Evaluation & monitoring

Evaluation and monitoring are critical components to understand if your agent application is performing to the *quality*, *cost*, and *latency* requirements dictated by your use case.  Technically, **evaluation** happens during development and **monitoring** happens once the application is deployed to production, but the fundamental components are similar.

Often, an agent is a complex system with many components that impact the application's quality. Adjusting any single element can have cascading effects on the others. For instance, data formatting changes can influence the retrieved chunks and the LLM's ability to generate relevant responses. Therefore, it's crucial to evaluate each of the application's components in addition to the application as a whole in order to iteratively refine it based on those assessments.

Evaluation and monitoring of Generative AI applications, including agents, differs from classical machine learning in several ways:
  
|  | Classical ML | Generative AI | 
|---------|---------|---------|
| **Metrics** | Metrics evaluate the __inputs & outputs__ of the component e.g., feature drift, precision/recall, latency, etc <br/><br/> Since there is only one component, overall metrics == component metrics. | __Component metrics__ evaluate the __inputs & outputs__ of each component e.g., precision @ K, nDCG, latency, toxicity, etc <br/><br/>__Compound metrics__ evaluate how multiple components interact e.g., faithfulness measures the generator’s adherence to the knowledge from a retriever which requires the chain input, chain output, and output of the internal retriever<br/><br/>__Overall metrics__ evaluate the overall input & output of the system e.g., answer correctness, latency |
| **Evaluation** | Answer is __deterministically__ “right” or “wrong” <br/><br/> → __Deterministic metrics__ work | Answer is “right” or “wrong” but: <br/><ul><li>Many right answers (non deterministic)</li><li>Some right answers are more right</li></ul><br/>→ Need __human feedback__ to be confident<br/>→ Need __LLM-judged metrics__ to scale evaluation<br/> |

Effectively evaluating and monitoring application quality, cost and latency requires several components:


```{image} ../images/2-fundamentals-unstructured/4_img.png
:align: center
```
<br/>

- **Evaluation set:** To rigorously evaluate your agent application, you need a curated set of evaluation queries (and ideally outputs) that are representative of the application's intended use. These evaluation examples should be challenging, diverse, and updated to reflect changing usage and requirements.

- **Metric definitions**: You can't manage what you don't measure. In order to improve agent quality, it is essential to define what quality means for your use case. Depending on the application, important metrics might include response accuracy, latency, cost, or ratings from key stakeholders.  You'll need metrics that measure each component, how the components interact with each other, and the overall system.

- **LLM judges**: Given the open ended nature of LLM responses, it is not feasible to read every single response each time you evaluate to determine if the output is correct.  Using an additional, different LLM to review outputs can help scale your evaluation and compute additional metrics such as the groundedness of a response to 1,000s of tokens of context, that would be infeasible for human raters to effectively asses at scale.

- **Evaluation harness**: During development, an evaluation harness helps you quickly execute your application for every record in your *evaluation set* and then run each output through your LLM judges and metric computations.  This is particularly challenging since this step "blocks" your inner dev loop, so speed is of the utmost importance.  A good evaluation harness will parallelize this work as much as possible, often spinning up additional infrastructure such as more LLM capacity to do so.

- **Stakeholder-facing UI:** As a developer, you may not be a domain expert in the content of the application you are developing. In order to collect feedback from human experts who can assess your application quality, you need an interface that allows them to interact with the application and provide detailed feedback.

- **Production trace logging:** Once in production, you'll need to evaluate a significantly higher quantity of requests/responses in addition to how each response was generated.  For example, you will need to  know if the root cause of a low quality answer is due to the retrieval step or a hallucination. Your production logging must track the inputs, outputs, and intermediate steps such as document retrieval to enable ongoing monitoring and early detection and diagnosis of issues that arise in production.

We will cover evaluation in much more detail in [Section 4: Evaluation](/nbs/4-evaluation).
