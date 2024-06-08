### Identify the root cause of quality issues

Retrieval and generation are the 2 primary buckets of root causes.  To identify the root cause to focus on first, use the output of the Mosaic AI Quality Lab's LLM judges that you ran in the previous steps.  You should start with the Root Cause that appears most frequently in your dataset.  

To determine the root cause, use the output of the following metrics:

| Metric name | Quality Lab metric ID | Description | More details |
|-----|-----|-----|-----|
| LLM-judged retrieval precision | `retrieval/llm_judged/ chunk_relevance/precision/average` | Is each retrieved chunk relevant to the input request? | [docs link]() | [cookbook link]() |
| LLM-judged groundedness | `response/llm_judged/ groundedness/rating/percentage` | Is the generated response is factually consistent with the retrieved context e.g., "grounded"? | [docs link]() | [cookbook link]() |
| LLM-judged correctness vs. the ground-truth answer | `response/llm_judged/ correctness/rating/percentage` | Is the agent’s generated response is factually accurate and semantically similar to the provided ground-truth response? | [docs link]() | [cookbook link]() |
| LLM-judged relevance to query | `response/llm_judged/ relevance_to_query/rating/percentage` | Is the response relevant to the input request? |[docs link]() | [cookbook link]() |

Note: If you have human labeled ground-truth for which document should be retrieved for each question, you can optionally replace `retrieval/llm_judged/chunk_relevance/precision/average` with the score for `retrieval/ground_truth/document_recall/average`.

#### If ground-truth responses are available


<table>
  
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


#### No ground-truth responses available

If you do not have ground-truth responses available, `response/llm_judged/correctness/` can't be computed.  Use this table instead.


<table>
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