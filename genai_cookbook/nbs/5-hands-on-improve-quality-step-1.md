### Identify the root cause of quality issues

**Expected time:** 60 minutes

**Overview**

Retrieval and generation are the 2 primary buckets of root causes.  To identify the root cause to focus on first, we will use the output of the Mosaic AI Quality Lab's LLM judges that you ran in the previous [step](./5-hands-on-evaluate-poc.md) to identify the root cause that most impacts your app's quality.

To determine the root cause, use the output of the following metrics:
- `retrieval/llm_judged/chunk_relevance/precision/average`
- `response/llm_judged/groundedness/rating/percentage`
- `response/llm_judged/correctness/rating/percentage`
- `response/llm_judged/relevance_to_query/rating/percentage`

If you have human labeled ground-truth for which document should be retrieved for each question, you can optionally replace `retrieval/llm_judged/chunk_relevance/precision/average` with the score for `retrieval/ground_truth/document_recall/average`.

**Requirements**

- Your Evaluation results for the POC are available in MLflow 
  - If you followed the previous step, this will be the case!
- All requirements from previous steps


**Instructions**

How you root cause your app depends on if your evaluation set contains the ground-truth responses to your questions - stored in `expected_response`.  If you have these available, use the first table below.  Otherwise, use the second table.

1. Open the `06_root_cause_poc_quality_issues` Notebook
2. Press Run All
3. Review the output tables to determine the most frequent root cause in your application and follow the steps linked below:
  - [Improve retriever](./5-hands-on-improve-quality-step-1-retrieval.md)
  - [Improve LLM](./5-hands-on-improve-quality-step-1-generation.md)

**_Root cause analysis with available ground truth_**

<table class="table">
  
  <tr>
   <td>Retrieval precision >50% 
   </td>
   <td>Groundedness
   </td>
   <td>Correctness
   </td>
   <td>Relevance to query
   </td>
   <td>Issue summary
   </td>
   <td>Root cause
   </td>
   <td>Overall Rating
   </td>
  </tr>
  <tr>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td><strong>Retrieval is poor.</strong>
   </td>
   <td>Improve Retriever
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>LLM generates relevant response, but <strong>retrieval is poor</strong> e.g., the LLM ignores retrieval and uses its training knowledge to answer.
   </td>
   <td>Improve Retriever
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>Either
   </td>
   <td><strong>Retrieval quality is poor</strong>, but LLM gets the answer correct regardless.
   </td>
   <td>Improve Retriever
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>Response is grounded in retrieval, but <strong>retrieval is poor</strong>.
   </td>
   <td>Improve Retriever
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>Relevant response grounded in the retrieved context, but <strong>retrieval may not be related to the expected answer.</strong>
   </td>
   <td>Improve Retriever
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>Either
   </td>
   <td>Retrieval is poor, but good enough for this question
   </td>
   <td>N/A
   </td>
   <td>✅
   </td>
  </tr>
  <tr>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>Either
   </td>
   <td>Hallucination
   </td>
   <td>Improve LLM
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>Either
   </td>
   <td>Hallucination, correct but generates details not in context
   </td>
   <td>Improve LLM
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>Good retrieval, but the LLM does not provide a relevant response.
   </td>
   <td>Improve LLM
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>Good retrieval and relevant response, but not correct.
   </td>
   <td>Improve LLM
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>No issue!! 🎉
   </td>
   <td>N/A
   </td>
   <td>✅
   </td>
  </tr>
</table>

<br/>
<br/>


**_Root cause analysis WITHOUT available ground truth_**

<table class="table">
  <tr>
   <td>Retrieval precision >50% 
   </td>
   <td>Groundedness
   </td>
   <td>Relevance to Query
   </td>
   <td>Issue summary
   </td>
   <td>Root cause
   </td>
   <td>Overall rating
   </td>
  </tr>
  <tr>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>Retrieval quality is poor
   </td>
   <td>Improve Retriever
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>Retrieval quality is poor
   </td>
   <td>Improve Retriever
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>Response is grounded in retrieval, but <strong>retrieval is poor</strong>.
   </td>
   <td>Improve Retriever
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>Relevant response grounded in the retrieved context and relevant, but <strong>retrieval is poor</strong>.
   </td>
   <td>Improve Retriever
   </td>
   <td>✅
   </td>
  </tr>
  <tr>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>Hallucination
   </td>
   <td>Improve LLM
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>Hallucination
   </td>
   <td>Improve LLM
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>Good retrieval & grounded, but LLM does not provide a relevant response.
   </td>
   <td>Improve LLM
   </td>
   <td>❌
   </td>
  </tr>
  <tr>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>Good retrieval and relevant response.  Collect ground-truth to know if the answer is correct.
   </td>
   <td>None
   </td>
   <td>✅
   </td>
  </tr>
</table>

<br/>
<br/>