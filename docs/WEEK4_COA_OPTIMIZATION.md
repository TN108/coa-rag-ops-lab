# Week 4: COA Optimisation

## Week 3 Baseline

- Retrieval hit rate: 0.9333
- MRR: 0.4655
- Answerable response rate: 0.2667
- Unanswerable accuracy: 1.0000
- Claim approval rate: 0.3158
- Rejected-claim leakage rate: 0.0000
- Average total latency: 48589 ms

## Failure Categories

### 1. Early answerability gate failures

Questions:
- q009
- q014
- q015

Problem:
Relevant evidence was retrieved, but reasoning was skipped in approximately
1 to 2 milliseconds.

### 2. Relevant evidence ranked too low

Questions:
- q002
- q008
- q010

Problem:
The correct evidence was ranked near position four or five, while the reasoning
agent may only inspect three chunks.

### 3. False critic rejections

Questions:
- q003
- q004
- q005
- q006
- q011
- q012
- q013

Problem:
The reasoning claims were relevant, but the critic rejected valid paraphrases
or required wording that was unnecessarily exact.

### 4. Weak answer relevance

Question:
- q007

Problem:
The selected claim was supported by the document but did not directly answer
how LangGraph implements multi-agent workflows.

## Week 4 Targets

- Retrieval hit rate >= 0.93
- MRR >= 0.55
- Answerable response rate >= 0.80
- Unanswerable accuracy = 1.00
- Claim approval rate >= 0.70
- Rejected-claim leakage rate = 0.00
- Reduce average latency substantially

## Day 1: Reasoning Context Selection

### Change

The reasoning agent now uses:

- 3 chunks for simple definition and authorship questions
- 5 chunks for explanation, architecture, routing and other complex questions

### Manual Test Results

#### q002: Main graph components

- Relevant evidence rank: 5
- Reasoning chunk limit: 5
- Relevant evidence reached reasoning: Yes
- Structured claims generated: 0
- Reasoning latency: 51592.76 ms
- Result: Reasoning extraction failure

#### q008: Role of shared state

- Relevant evidence rank: 5
- Reasoning chunk limit: 5
- Relevant evidence reached reasoning: Yes
- Structured claims generated: 0
- Reasoning latency: 33524.74 ms
- Result: Reasoning extraction failure

#### q010: Purpose of StateGraph

- Relevant evidence rank: 2
- Reasoning chunk limit: 5
- Relevant evidence reached reasoning: Yes
- Structured claims generated: 0
- Reasoning latency: 51997.05 ms
- Result: Reasoning extraction failure

### Day 1 Conclusion

Expanding the reasoning context successfully allowed lower-ranked evidence at
rank five to reach the reasoning agent for q002 and q008.

However, all three questions still produced zero structured claims. The primary
remaining failure is therefore structured claim extraction by the reasoning
model, not missing evidence or critic rejection.

No critic, answerability-gate, retrieval-ranking or synthesis changes were made.
