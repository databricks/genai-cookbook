### **Step 5:** Identify the root cause of quality issues

```{image} ../images/5-hands-on/workflow_iterate.png
:align: center
```
<br/>

**Expected time:** 60 minutes

#### **Requirements**

1. Your Evaluation results for the POC are available in MLflow 
    - If you followed the previous step, this will be the case!
2. All requirements from previous steps

#### **Overview**

Retrieval and generation are the 2 primary buckets of root causes.  To determine where we focus on first, we  use the output of the Mosaic AI Agent Evaluation's LLM judges that you ran in the previous [step](./5-hands-on-evaluate-poc.md) to identify the most frequent root cause that impacts your app's quality.  W


Each row your evaluation set will be tagged as follows:
1. **Overall assessment:** ![pass](../images/5-hands-on/pass.png) or ![fail](../images/5-hands-on/fail.png)
2. **Root cause:** `Improve Retrieval` or `Improve Generation`
3. **Root cause rationale:** A brief description of why the root cause was selected

#### **Instructions**

The approach depends on if your evaluation set contains the ground-truth responses to your questions - stored in `expected_response`.  If you have `expected_response` available, use the first table below.  Otherwise, use the second table.

1. Open the `B_quality_iteration/01_root_cause_quality_issues` Notebook
2. Run the cells that are relevant to your use case e.g., if you do or don't have `expected_response`
3. Review the output tables to determine the most frequent root cause in your application
4. For each root cause, follow the steps below to further debug and identify potential fixes:
    - [Debugging retrieval quality](./5-hands-on-improve-quality-step-1-retrieval.md)
    - [Debugging generation quality](./5-hands-on-improve-quality-step-1-generation.md)

##### Root cause analysis _with_ available ground truth

```{note}
If you have human labeled ground-truth for which document should be retrieved for each question, you can optionally substitute `retrieval/llm_judged/chunk_relevance/precision/average` with the score for `retrieval/ground_truth/document_recall/average`.
```

<table class="table">
  
  <tr>
   <td>Chunk relevance precision
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
   <td><div style="color:red">&lt;50%</div>
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td><strong>Retrieval is poor.</strong>
   </td>
   <td><code>Improve Retrieval</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:red">&lt;50%</div>
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>LLM generates relevant response, but <strong>retrieval is poor</strong> e.g., the LLM ignores retrieval and uses its training knowledge to answer.
   </td>
   <td><code>Improve Retrieval</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:red">&lt;50%</div>
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>✅ or ❌
   </td>
   <td><strong>Retrieval quality is poor</strong>, but LLM gets the answer correct regardless.
   </td>
   <td><code>Improve Retrieval</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:red">&lt;50%</div>
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>Response is grounded in retrieval, but <strong>retrieval is poor</strong>.
   </td>
   <td><code>Improve Retrieval</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:red">&lt;50%</div>
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>Relevant response grounded in the retrieved context, but <strong>retrieval may not be related to the expected answer.</strong>
   </td>
   <td><code>Improve Retrieval</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:red">&lt;50%</div>
   </td>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>✅ or ❌
   </td>
   <td>Retrieval finds enough information for the LLM to correctly answer. 🎉
   </td>
   <td>N/A
   </td>
   <td><img src="../_images/pass.png" alt="pass" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:green">&gt;50%</div>
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>✅ or ❌
   </td>
   <td>Hallucination
   </td>
   <td><code>Improve Generation</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:green">&gt;50%</div>
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>✅ or ❌
   </td>
   <td>Hallucination, correct but generates details not in context
   </td>
   <td><code>Improve Generation</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:green">&gt;50%</div>
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>Good retrieval, but the LLM does not provide a relevant response.
   </td>
   <td><code>Improve Generation</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:green">&gt;50%</div>
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>Good retrieval and relevant response, but not correct.
   </td>
   <td><code>Improve Generation</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:green">&gt;50%</div>
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
   <td><img src="../_images/pass.png" alt="pass" height="20"/> 
   </td>
  </tr>
</table>

<br/>
<br/>


##### Root cause analysis _without_ available ground truth

<table class="table">
  <tr>
   <td>Chunk relevance precision
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
   <td><div style="color:red">&lt;50%</div>
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>Retrieval quality is poor
   </td>
   <td><code>Improve Retrieval</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:red">&lt;50%</div>
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>Retrieval quality is poor
   </td>
   <td><code>Improve Retrieval</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:red">&lt;50%</div>
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>Response is grounded in retrieval, but <strong>retrieval is poor</strong>.
   </td>
   <td><code>Improve Retrieval</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:red">&lt;50%</div>
   </td>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>Relevant response grounded in the retrieved context and relevant, but <strong>retrieval is poor</strong>.
   </td>
   <td><code>Improve Retrieval</code>
   </td>
   <td><img src="../_images/pass.png" alt="pass" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:green">&gt;50%</div>
   </td>
   <td>❌
   </td>
   <td>❌
   </td>
   <td>Hallucination
   </td>
   <td><code>Improve Generation</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:green">&gt;50%</div>
   </td>
   <td>❌
   </td>
   <td>✅
   </td>
   <td>Hallucination
   </td>
   <td><code>Improve Generation</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:green">&gt;50%</div>
   </td>
   <td>✅
   </td>
   <td>❌
   </td>
   <td>Good retrieval & grounded, but LLM does not provide a relevant response.
   </td>
   <td><code>Improve Generation</code>
   </td>
   <td><img src="../_images/fail.png" alt="fail" height="20"/> 
   </td>
  </tr>
  <tr>
   <td><div style="color:green">&gt;50%</div>
   </td>
   <td>✅
   </td>
   <td>✅
   </td>
   <td>Good retrieval and relevant response.  Collect ground-truth to know if the answer is correct.
   </td>
   <td>None
   </td>
   <td><img src="../_images/pass.png" alt="pass" height="20"/> 
   </td>
  </tr>
</table>

<br/>
<br/>