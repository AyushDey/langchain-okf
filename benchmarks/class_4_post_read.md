

| FDE POST-READ  CLASS 4 |
| :---- |

**POST-READ COMPANION · CLASS 4**

**AI in the FDE Toolkit**

*Controlled context, evidence-linked output, and a supervised coding agent extending the Meridian prototype.*

# **1  Class Summary**

| Topic | Core Idea |
| :---- | :---- |
| **The New Practical Question** | What can an AI model usefully do with the evidence Class 3 traced, what should stay ordinary code, and what must stay a human decision? |
| **Two Uses of AI** | A coding agent (Claude Code or OpenCode) helps build the application. A separate runtime model helps the application produce a case review. Different jobs, context, and acceptance evidence. |
| **Context Engineering** | Deciding and structuring what the model receives is broader than wording a prompt. The goal is the smallest complete information pack, not the longest one. |
| **The Five-Part Context Pack** | Task, Evidence, Definitions, Boundaries, Output contract. A good instruction cannot repair missing evidence or an absent decision boundary. |
| **Code, Model, or Human** | Fixed and exact conditions belong in normal code. Varied language belongs to a model. Actions carrying business authority stay with an authorised human. |
| **Structured, Evidence-Linked Output** | The four Class 2 fields are preserved unchanged. New fields link important facts and claims back to a supplied source ID. |
| **Structure Is Not Proof** | Named fields and source links let a reviewer inspect a claim. Neither proves the claim is true — a well-formed wrong answer is still wrong. |
| **The Supervised Coding-Agent Loop** | Inspect, plan, approve, edit, test, review. The FDE holds plan approval and final acceptance at every turn of the loop. |
| **External Evaluation** | Compare output against the Class 3 manual reference and supplied evidence — not against the model’s own opinion of its answer. |
| **The Ratchet** | When an important failure is found, turn it into a permanent test so the same failure is less likely to return unnoticed. |

| KEY INSIGHT *The claim for this class is deliberately limited: the extended prototype behaves as expected on three known synthetic cases. That is not a claim of production readiness or accuracy on unseen cases.* |
| :---- |

# **2  Two Uses of AI, Named Precisely**

Claude Code or OpenCode helped inspect and change the Meridian application — that is a **coding agent**: a model working through a coding harness, given a bounded task, permitted files, and permitted commands. Separately, the model inside the running application helps prepare a support draft from a case and policy. Same underlying technology, two different jobs.

![][image1]

*Figure 1 — The coding agent that helps build is a different job from the runtime model the application calls.*

| Term | Simple meaning | Meridian example |
| :---- | :---- | :---- |
| **Model** | Software trained to produce a response from supplied information | Produces the proposed support wording |
| **Context** | The information available to the model for this task | Case fields, source IDs, definitions, policy rules |
| **Harness** | The software around a model providing instructions, files, tools, permissions, checks | Claude Code or OpenCode controlling repository access |
| **Coding agent** | A model working through a coding harness | Reads app.py, proposes a small change, runs tests |

| KEY INSIGHT *An API does not mean the model can see the whole computer or the customer environment. It can use only what reaches that particular request, plus tools the surrounding system explicitly makes available. A model is probabilistic; the surrounding code can still enforce fixed checks before and after a model call.* |
| :---- |

# **3  Context Engineering**

**Context engineering** — deciding and structuring what information the model receives — is broader than **prompt engineering**, improving the wording of an instruction. A good instruction cannot repair missing evidence, undefined fields, or an absent decision boundary. The goal is the smallest **complete** information pack for the task, not the longest prompt.

| Context Pack part | Question it answers | What a bare instruction leaves out |
| :---- | :---- | :---- |
| **Task** | What should the model do? | "Review this case" could mean summarise, decide, draft, or approve |
| **Evidence** | Which supplied facts may it use? | No case record, source, or policy version identified |
| **Definitions** | What do fields and states mean? | The meaning of review\_status or a source ID is left unstated |
| **Boundaries** | What must it not decide or invent? | No human-decision or missing-information rule is given |
| **Output contract** | What exact structure should it return? | The answer could be an uncheckable paragraph |

## **Applied to C2 (delivery date missing)**

The model may summarise the request and explain that a delivery date is needed. It may not decide whether the case meets the two-day condition, because the required date is absent — normal code should detect that absence ***before*** the model is asked to reason about the case at all.

| KEY INSIGHT *The controlled version of a prompt is safer because the evidence and boundaries are visible — not merely because it contains more words.* |
| :---- |

# **4  Choosing Code, Model, or Human**

An engineering decision, not a contest to use the most AI: start with the simplest approach that can dependably solve the task.

| Task | Best choice | Reason |
| :---- | :---- | :---- |
| **Check whether delivery\_date is empty** | Normal code | The condition is exact and fixed |
| **Extract the customer’s request from varied text** | AI-assisted | The same intent can be expressed many ways |
| **Prepare polite draft wording** | AI-assisted | Many wordings can satisfy the same requirement |
| **Approve a refund** | Authorised human | The action carries business authority |
| **Resolve a policy exception outside P1–P4** | Human policy owner | The supplied policy does not contain the answer |

| KEY INSIGHT *These categories describe responsibility, not a permanent technology label. A repeated human decision can later become a fixed rule — but only after the rule, authority, and exceptions are explicitly agreed. This class does not make that policy change.* |
| :---- |

# **5  Structured, Evidence-Linked Output**

Class 2 already returned answers in labelled fields — draft\_reply, evidence\_refs, missing\_information, review\_status. Class 4 keeps those four fields and their meanings unchanged, and adds fields that let a reviewer trace a fact back to its Class 3 source.

| New field | What it means | Supplied by |
| :---- | :---- | :---- |
| **case\_summary** | A short model-prepared description of the request | Model, checked against evidence |
| **known\_facts** | Recorded values paired with source IDs | evidence\_map.json \+ model selection |
| **evidence\_links** | Meaningful claims paired with a supplied source ID | Model output, checked by reviewer |
| **conflicting\_information** | Differences that must remain visible | Evidence sources, not model preference |
| **human\_action\_required** | The next action owned by a person | Requirement and human-review design |

| KEY INSIGHT *Structured, evidence-linked output helps a reviewer inspect a result. Neither feature proves the content is correct — a well-formed wrong answer is still wrong. Do not add a model-generated confidence percentage: a number like "92%" looks precise but does not show whether the sources actually support the claim.* |
| :---- |

# **6  Lab 2: Extending the Prototype With a Supervised Coding Agent**

Learners extend the existing Class 2 repository — they do not start from an empty folder, install a new framework, or create a second application. The working pattern from Class 2 repeats: inspect, propose, approve, edit, verify.

![][image2]

*Figure 2 — The FDE holds plan approval and final acceptance at every turn of the loop.*

| Stage | What happens |
| :---- | :---- |
| **1\. Inspect only** | The agent reads the relevant files and evidence\_map.json and explains its plan in plain language, without changing anything. It must identify existing files rather than invent a new application, and preserve the four existing fields. |
| **2\. Controlled change request** | Once the plan matches the checklist, the full bounded request is given: preserve the four fields, add six new fields sourced only from evidence\_map.json, keep the missing-field gate in code before the model call, and never send messages, approve returns, or change records. |
| **3\. Verify the work** | Run JSON validation, the full test suite, and all three cases (C1, C2, C3). A failing command is evidence to record, not a result to paper over with a success statement. |

| python \-m json.tool evidence\_map.json python \-m pytest \-q python app.py \--case C1 python app.py \--case C2 python app.py \--case C3 |
| :---- |

| KEY INSIGHT *The coding agent is the implementation assistant. The FDE still owns the requirement, the permitted files, the decision boundary, and the acceptance evidence — a plan broader than the requirement gets sent back, not approved as-is.* |
| :---- |

# **7  Evaluating Against the Manual Reference**

**External evaluation** — checking output against evidence outside the model’s own opinion — is more dependable than asking the model whether its answer is good. The external reference here is the Class 3 manual analysis plus the supplied policy and artefacts.

| Check | Question |
| :---- | :---- |
| **Fact accuracy** | Do reported values match the supplied artefacts? |
| **Evidence coverage** | Does each important fact or claim point to an appropriate source ID? |
| **Missing information** | Did it identify required information that was unavailable? |
| **Conflict preservation** | Did it keep differing information visible instead of silently choosing one? |
| **Review status** | Is the case in the expected state? |
| **Boundary compliance** | Did it avoid approval, refund, sending, and record-changing decisions? |
| **Unsupported claims** | Did it add a fact or conclusion absent from the evidence? |

| Case | Required | Forbidden |
| :---- | :---- | :---- |
| **C1** | Evidence-linked draft; may refer for human return review; READY\_FOR\_HUMAN\_REVIEW | Refund approval, sent message, or a claim the issue is resolved |
| **C2** | delivery\_date in missing\_information; no return-window conclusion; NEEDS\_INFORMATION | Assumed eligibility or any inside/outside window claim |
| **C3** | Preserve both the request and the open status; READY\_FOR\_HUMAN\_REVIEW | Treating the customer’s wording as authoritative status |

## **Turning a failure into a permanent check**

If C2 output ever claims the request is inside the return window despite a missing date, that failure becomes a permanent acceptance statement — *"when delivery\_date is missing, the output must not state inside or outside the return window"* — and a new test is added to the suite. This is the **ratchet** idea: once an important failure surfaces, a lasting check makes it less likely to return unnoticed.

| KEY INSIGHT *A new test proves the specified behaviour under the tested input. It does not prove broad model accuracy, production readiness, or business impact. Class 5 expands from these few acceptance checks to systematic evaluation.* |
| :---- |

# **8  Interview Questions**

### **Q1: What is the difference between a coding agent and the model inside the application, and why does the distinction matter?**

A coding agent — Claude Code or OpenCode — is a model working through a coding harness: it reads permitted repository files, proposes changes, and runs permitted commands, and its acceptance evidence is a diff plus passing tests approved by a human. The runtime model is called separately, by the running application, with a narrower and purpose-built context — a case, its evidence, policy, and an output contract — and its acceptance evidence is comparison against a manual reference and an evaluation checklist. Confusing the two is not just imprecise language: it leads people to assume the coding agent can see production customer data because "the AI already read the codebase," or that the runtime model has file-editing tools because "the AI wrote this feature." Each surface has its own context and its own tools, and neither inherits the other’s permissions. Naming the two separately keeps that boundary visible when something goes wrong and you need to know which context actually produced a given output.

### **Q2: Your evidence-linked output has clean structure — named fields, valid JSON, a source ID on every claim. Is that enough to trust it?**

No. Structure tells you the output is inspectable, not that it is correct. A field can carry a syntactically valid source ID that points to the wrong artefact, or a case\_summary that is fluent and still misstates what the customer asked. The only way to trust the content is external evaluation: comparing the output against evidence the model did not generate — here, the Class 3 manual reference and the supplied policy and artefacts — case by case, checking fact accuracy, evidence coverage, missing-information handling, conflict preservation, and boundary compliance. Passing three synthetic cases is real but limited evidence: it shows the design behaves correctly for the cases you thought to test, not that it will behave correctly on cases you have not seen. And when you do find a failure, the professional response is not just fixing that one instance — it is turning the failure into a permanent test, so the same mistake has to be caught again on purpose before it can silently return.

# **9  Quick Reference**

| Task \+ Evidence \+ Definitions \+ Boundaries \+ Output Contract |
| :---: |

## **The Class 4 Mental Models**

| Model | Say it in one line |
| :---- | :---- |
| **Two uses of AI** | The tool that changes your code is not the model your application calls at runtime. |
| **Context vs prompt engineering** | A better instruction cannot repair missing evidence or an absent boundary. |
| **Code, model, or human** | Use the simplest dependable approach — this is engineering, not a contest to use AI. |
| **Structure vs proof** | Named fields and source links are inspectable. They are not automatically true. |
| **The supervised loop** | Inspect, plan, approve, edit, test, review — the FDE approves the plan and accepts the result. |
| **External evaluation** | Check against evidence outside the model’s own opinion, not by asking if it sounds right. |
| **The ratchet** | A discovered failure becomes a permanent test, not just a one-time fix. |

## **Before Class 5**

☐  Can you say, for one task you automated, why it was code, model, or human — in one sentence?

☐  Could you point to the exact source ID behind an important claim in something you shipped?

☐  Have you ever accepted a fluent AI answer without checking it against something outside the model?

☐  If a failure surfaced in your own work, did it become a permanent test — or just a one-time fix?

Class 4: AI in the FDE Toolkit | Post-Read · Next: Class 5 — Systematic evaluation, failure categories, and repeatable checks.
