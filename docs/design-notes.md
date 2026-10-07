# Design notes

## Main pipeline

1. Index the documents.
2. Classify the intent using Jev. Jev handles:
   - intent
   - retrieved articles, with the help of keyword search + reranking: top 3 from each search (6 in total), then the 3 best results
   - confidence
   - safety flags
3. The LLM generates the reply based on what Jev has returned.
4. Validate the output using a Pydantic `BaseModel`.
5. If the results have the correct structure, write them to `results.json`.

## Offline

- Use cosine similarity to compare the ticket to our documents and classify the intent.
- Only take high-confidence scores (> 0.8) and put the others in a queue for the pipeline to classify when the device
  is online again.

## Confidence and escalation

A low confidence score from the Jev model should escalate to a human, e.g. for:

- a low retrieval score
- low confidence in the relevant articles
