# Optimizing RLMs

Recursive Language Models were introduced by Alex Zhang et. al in [https://arxiv.org/abs/2512.24601](https://arxiv.org/abs/2512.24601). At its core, it introduces a scaffolding where the language model is able to iteratively interact with a repl and call itself recursively for long-context tasks. 

This repo is a re-implementation from scratch in Go. Our Objective: Optimize speed and accuracy on the oolong Pairs benchmark as much as possible. We won't be so concerned about having modular components in our code to handle different base llms/environments/etc. We will stick with Docker and GPT5 with reasoning effort set to medium. 

## How [the original RLM](https://github.com/alexzhang13/rlm) Works

At a high level:

1. **A task comes in** — for OOLong, this is a long document (16k or 65k tokens) plus a natural-language query asking the model to find pairs of entries matching some criteria.
2. **The RLM loop begins** — the LLM receives the context and a system prompt describing the tools available to it: a Python REPL running inside a Docker container, the ability to query itself (`llm_query`), and the ability to spawn child RLMs (`rlm_query`) that run their own independent iteration loops.
3. **Each iteration**, the model can write and execute Python code in the REPL, call sub-LLMs or sub-RLMs, and inspect results. We pass on the message history and execution results of previous iterations as well
4. **Termination** — the model signals completion by calling `FINAL(answer)` or `FINAL_VAR(variable_name)` in its Python code, and the answer is extracted and returned.

This repo was written completely in Python. For us, we will use Go to write the scaffolding/orchestration ( managing the Docker container lifecycle, parsing code blocks out of LLM responses, routing HTTP callbacks from the REPL back to the LLM, and enforcing iteration/depth/timeout limits) yet will still stick to Python for the llm to use. This is done for 1. consistency for comparison purposes and 2. the belief that models in general are better in Python due to abundant training data. 

## OOLong Pairs Benchmark

The OOLong-Pairs benchmark (Appendix D.1 of the paper) asks the model to find pairs of questions in a large corpus (i.e. "find all pairs where one question is about X and the other is about Y") sometimes with asymmetric constraints, exact count requirements, or date filters, etc.

There are 20 tasks total. Running and evaluating for these tasks even when the dataset is small takes quite a while, so we just focus on a small subset of the tasks that the RLM struggles on the most. 

### Baseline Results (Python RLM, 16k, gpt-5 default reasoning, 20 iters)

The paper only reports average F1 across all 20 tasks. We re-ran the Python reference code on tasks that it struggled in the most (Task 3 is included because it often resulted in 0.0 F1 but this run happens to have been a luckier one). Max iterations was set to 20 instead of the default 10.  


| Task | Gold | Pred | P    | R    | F1   | Time |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 3    | 1891 | 1953 | 0.94 | 0.97 | 0.95 | 668s |
| 5    | 120  | 276  | 0.38 | 0.88 | 0.53 | 438s |
| 7    | 1830 | 861  | 0.91 | 0.43 | 0.58 | 247s |
| 9    | 1653 | 861  | 0.95 | 0.50 | 0.65 | 478s |
| 11   | 461  | 510  | 0.78 | 0.86 | 0.82 | 445s |
| 16   | 85   | 100  | 0.65 | 0.76 | 0.70 | 321s |
| 19   | 42   | 54   | 0.70 | 0.90 | 0.79 | 268s |


Average F1 over these 7 tasks: **0.72**. Total wall clock: ~48 minutes (one run). 

**A caveat on variance:** LLM outputs are inherently non-deterministic, and RLM amplifies that since the model's code and reasoning can diverge significantly between runs. We found that variance is very high and sometimes the results differ by a magnitude. Take the exact numbers with a grain of salt.

## Workflow Decisions

1. Initial Attempt:  History compaction (`compacted-history` branch)

Initially, I spent a lot of time trying to take most of the existing structure of the RLM repository and iterate upon it. Managing the history of this conversation was certainly interesting. I tried a bunch of stuff. 

Inspired by [https://arxiv.org/pdf/2602.24287](https://arxiv.org/pdf/2602.24287) - the answers given by llm itself may not always be the best. Probably needed when llm needs to refer to its own answer. We keep a summary, then we keep the full text as a variable that the rlm can access. 

In practice, this means the compact pipeline doesn't send the full conversation history back to the LLM on each turn. Instead, it maintains a **context ledger** — a concise summary of what happened in each prior turn. The full outputs are stored as REPL variables (`_turn_N_response`, `_turn_N_output`) that the model can `print()` on demand if it needs to revisit something. This keeps the prompt size manageable as the iteration count grows. 

Results (Go RLM Compact, 16k, gpt-5 medium, 20 iters):


| Task | Gold | Run 1 F1 | Run 1 Time | Run 2 F1 | Run 2 Time |
| ---- | ---- | -------- | ---------- | -------- | ---------- |
| 3    | 1891 | 0.58     | 352s       | 0.97     | 146s       |
| 5    | 120  | 0.54     | 446s       | 0.00     | 397s       |
| 7    | 1830 | 0.56     | 837s       | 0.56     | 248s       |
| 9    | 1653 | 0.65     | 398s       | 0.63     | 241s       |
| 11   | 461  | 0.77     | 321s       | 0.80     | 504s       |
| 16   | 85   | 0.59     | 165s       | 0.64     | 328s       |
| 19   | 42   | 0.50     | 155s       | 0.64     | 261s       |


Macro F1: **0.60** (run 1), **0.61** (run 2). Wall clock: 35/45min.

In summary, this workflow was not as powerful as the original workflow (though the average was brought down by task 5, and again, high variance). We oberseved from the traces of the RLM that it would often just rely on the summarized version instead of asking for the past history again. The run-time improved a lot though. We tried a bunch of stuff like this but it ended up more or less like this. 

2. My Approach: Truly recursive, single-shot execution (`main` branch)

The original Go implementation (like the Python reference) used a fixed iteration loop — the system called the LLM up to N times (like a chat), feeding back REPL output each round. The `truly-recursive-rlm` branch replaced this with **model-driven recursion**: the LLM is called once, produces a single code block, and if it needs more reasoning it calls `rlm_query()` from within its code, which recursively invokes another `Completion`. In other words, an RLM call must produce an answer with one iteration instead of allowing the model to relax back on the later iterations. This is a much more natural and truly recursive structure than before. Also, we don't need to pass on message history, the rlm naturally delegates relevant information when calling the rlm from within. 

Key changes:

- **Removed `maxIterations`** — We only iterate when the rlm fails to provide an answer in which case we provide the error message and then ask the rlm to one-shot it again. The model gets up to 2 retry attempts with the error context before giving up.
- **Rewrote the system prompt** The system prompt is our only means to change the behavior of the RLM. We added some concrete multi-turn examples and stronger `FINAL()` guidance.

Results (16k, gpt-5 medium, 7 hard tasks, averaged over 2 runs):


| Task    | F1         | Time        |
| ------- | ---------- | ----------- |
| 3       | 0.9678     | 213.8s      |
| 5       | 0.8394     | 178.8s      |
| 7       | 0.9628     | 151.1s      |
| 9       | 0.9054     | 182.2s      |
| 11      | 0.7257     | 200.0s      |
| 16      | 0.5494     | 226.8s      |
| 19      | 0.7833     | 205.1s      |
| **ALL** | **0.8191** | **1357.8s** |


A massive jump from the compact pipeline's 0.60 F1 and the Python reference implementation's 0.72. The orchestrator prompt and model-driven recursion let the model structure its approach much more effectively, especially on the harder tasks. Also, this is 2x faster than before. 

## Other Minor Changes

1. REPL Cold-Turkey Start on the REPL - Inefficient to start this again every thing we call on a new rlm repl. We don't really need independence for reproducibility - we just need a docker container where we can run code.
2. Streaming - Just send off tasks and run the python code as you go - don't wait.
3. Batching for Parallelism.

The batching point deserves some expansion: the system exposes `llm_query_batched` and `rlm_query_batched` tools that let the model fire off multiple LLM/RLM calls in parallel from a single Python code block. For tasks that involve classifying many documents independently, this can dramatically cut wall-clock time by overlapping API calls instead of making them sequentially.

Combined effect of these micro-changes on top of the truly recursive architecture (16k, gpt-5 medium, 7 hard tasks, averaged over 2 runs):


| Task    | Before F1  | Before Time | After F1   | After Time  |
| ------- | ---------- | ----------- | ---------- | ----------- |
| 3       | 0.9678     | 213.8s      | 0.9759     | 140.9s      |
| 5       | 0.8394     | 178.8s      | 0.8609     | 237.7s      |
| 7       | 0.9628     | 151.1s      | 0.9628     | 175.6s      |
| 9       | 0.9054     | 182.2s      | 0.9518     | 206.1s      |
| 11      | 0.7257     | 200.0s      | 0.7688     | 173.4s      |
| 16      | 0.5494     | 226.8s      | 0.5953     | 197.0s      |
| 19      | 0.7833     | 205.1s      | 0.7292     | 155.4s      |
| **ALL** | **0.8191** | **1357.8s** | **0.8350** | **1286.1s** |


The micro-changes cuts total wall clock by ~5%. The gains are most visible on tasks 9, 11, and 16 where REPL reuse and better parallelization reduce overhead. 

## Long Context Results (65k, ~1585 records)

Running the same truly-recursive architecture on the 65k dataset (4x larger context, ~1585 records across 490 users):


| Task    | Gold   | Pred   | F1        | Time      |
| ------- | ------ | ------ | --------- | --------- |
| 3       | 38,503 | 39,903 | 0.968     | 303s      |
| 5       | 6,670  | 5,565  | 0.793     | 478s      |
| 7       | 19,306 | 0      | 0.000     | 626s      |
| 9       | 9,045  | 11,476 | 0.856     | 452s      |
| 11      | 2,076  | 4,737  | 0.486     | 404s      |
| 16      | 530    | 0      | 0.000     | 269s      |
| 19      | 579    | 864    | 0.654     | 387s      |
| **ALL** |        |        | **0.537** | **48.7m** |


The 65k dataset is substantially harder. Tasks 7 and 16 produced zero predictions — the model either failed to parse the larger context or ran into issues completing the classification pipeline within the token budget. The paper reports a score of 0.60 (and remember, this is all 20 tasks, and the easier ones you can get a score of > 0.9, so our results are actually pretty good here). However, when we tested this, the model failed often. In the paper, they use the mini model for the subcalls but we always stuck with gpt5-medium model. This just goes to show how recursive and out-of-the-box our architecture is. 

## Directions I Did Not Pursue for the sake of This Project:

1. **Finetuning an LLM to use this scaffolding.** All the work we did was through prompts - but finetuning these language models to behave naturally like rlms is probably way more powerful! (This also means no CUDA stuff as we will just be making API Calls). No prefill/cache stuff as well.
2. **Switching Models at different depths of calling RLMs.** It makes sense to use smaller models for more straightforward tasks while reserving the powerful models for the high level orchestrating tasks. There are a lot of combinations possible. I did not experiment with this and just always stuck to GPT5 medium reasoning.



## Some Personal Learnings: 

1. Prompting is very important - the only way to change model behavior without getting into fine-tuning. 
2. Figure out cleaner/less variable evals - tough that temperature = 0 is not available for gpt-5. 
3. Keep better track of work progress - messy and had to rerun and redo a lot of things... 
4. Using Go wasn't really necessary - didn't really deal to much in concurrency - main bottlenecks were in the overall algorithm/api calls. 


