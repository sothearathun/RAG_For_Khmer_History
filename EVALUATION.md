# Evaluation

## Updated-code rerun — 22 July 2026

The full 10-query suite was rerun after adding coverage-oriented retrieval for
broad prompts. The standard-query results remain **9/10 fully grounded/correct**:
all nine in-domain queries retrieved their expected document in the top three,
query 1 retained its weaker-but-correct tangential citation, query 9 correctly
declined to invent an unsupported cause, and the out-of-domain France query was
refused by the 0.45 relevance gate (top score: 0.352).

The new behavior was tested separately with **"give me all important events
during Khmer Empire"**. It selected eight relevance-and-diversity-oriented
chunks, including a *Khmer Empire* source, and produced a grounded LLM answer
instead of declining the broad request. This confirms that fact questions still
use the regular top-k path while timelines, summaries, explanations, and
"all important events" prompts now receive broader non-duplicated context.

## Rerun — 22 July 2026

Step 7 was rerun against the current full 21-document corpus with
`all-MiniLM-L6-v2` retrieval (`top_k=3`) and DeepSeek `deepseek-v4-flash` in
`llm` mode. All nine in-domain queries retrieved their expected document in the
top three. The out-of-domain query was correctly rejected before an LLM request
by the 0.45 similarity floor.

| # | Query | Expected source retrieved in top-3? | Grounded/correct result |
|---|---|---|---|
| 1 | Who led the Khmer Rouge? | Yes — Democratic Kampuchea (0.707) | Partially grounded: correct leadership answer, but it cites the tangential Cambodian-Vietnamese War chunk rather than Democratic Kampuchea. |
| 2 | When did Cambodia gain independence from France? | Yes — French Protectorate Of Cambodia (0.733) | Yes — 9 November 1953. |
| 3 | Who was Norodom Sihanouk? | Yes — Norodom Sihanouk (0.675) | Yes — accurate summary and source citation. |
| 4 | What happened during the 1997 Cambodian coup? | Yes — 1997 Cambodian coup d'état (0.731) | Yes — correctly identifies Hun Sen, Norodom Ranariddh, and the factional fighting. |
| 5 | What was Funan? | Yes — Funan (0.615) | Yes — correctly describes the polity, Chenla's absorption, and the cited historical evidence. |
| 6 | What was the Khmer Empire known for? | Yes — Khmer Empire (0.733) | Yes — hydraulic cities and Angkor. |
| 7 | What happened during the Cambodian genocide? | Yes — Cambodian Genocide (0.700) | Yes — correct period, perpetrators, scale, and targeted groups. |
| 8 | What was UNTAC's role in Cambodia? | Yes — United Nations Administered Cambodia (0.758) and United Nations Transitional Authority (0.757) | Yes — correctly describes its 1992–93 administration, peacekeeping, human-rights, and election roles. |
| 9 | What caused the Cambodian Civil War? | Yes — Cambodian Civil War (0.697) | Yes — correctly declines to invent a cause not supported by the retrieved passages. |
| 10 | What is the capital of France? | N/A — no relevant source; top score 0.352 | Yes — declined without calling the LLM because the score is below 0.45. |

**Rerun score: 9/10 fully grounded/correct, 1/10 partially grounded due to a
weaker-than-ideal citation choice (query 1).** The retrieval and abstention
behavior are working as intended; the remaining issue is citation selection
among multiple relevant chunks.

10 test queries run against the live pipeline (`sentence-transformers` retrieval, top_k=3, Gemini `gemini-2.5-flash` in `llm` mode) on the full 21-document corpus. For each: the expected source(s), what was actually retrieved, and whether the generated answer was grounded and correct.

| # | Query | Expected source(s) | Retrieved top-3 | Retrieval hit? | Answer grounded/correct? |
|---|---|---|---|---|---|
| 1 | Who led the Khmer Rouge? | Democratic Kampuchea | Democratic Kampuchea, Cambodian-Vietnamese War, Khmer Rouge Insurgency | Partial | Correct (Pol Pot, Ieng Sary, Khieu Samphan), but cited to a tangential quote in *Cambodian-Vietnamese War* rather than a clean statement — see note below |
| 2 | When did Cambodia gain independence from France? | French Protectorate of Cambodia | French Protectorate of Cambodia (all 3) | Yes | Yes — "9 November 1953", correctly cited |
| 3 | Who was Norodom Sihanouk? | Norodom Sihanouk | Norodom Sihanouk (all 3) | Yes | Yes — accurate, multi-paragraph, each claim cited |
| 4 | What happened during the 1997 Cambodian coup? | 1997 Cambodian coup d'état | 1997 Cambodian coup d'état, History of Cambodia (1993-present), Khmer Rouge Insurgency | Yes | Yes — correctly synthesized across all 3 sources, dates and actors correct |
| 5 | What was Funan? | Funan | Funan (all 3) | Yes | Yes — correct kings, correct absorption by Chenla |
| 6 | What was the Khmer Empire known for? | Khmer Empire | Khmer Empire, Chenla, Khmer Empire | Yes | Yes — correct (hydraulic cities, Angkor) |
| 7 | What happened during the Cambodian genocide? | Cambodian Genocide | Cambodian Genocide (all 3) | Yes | Yes — correct death toll (~2M, ~25% of population), correct actors |
| 8 | What was UNTAC's role in Cambodia? | United Nations Transitional Authority in Cambodia | United Nations Administered Cambodia, United Nations Transitional Authority, History of Cambodia (1993-present) | Yes | Mostly — correct content (elections, human rights mandate) but the answer's prose is choppy, and one retrieved chunk's "section" label is corpus noise (see note below) |
| 9 | What caused the Cambodian Civil War? | Cambodian Civil War | Cambodian Civil War (all 3) | Yes | Yes — correctly says the sources don't state an explicit cause rather than inventing one |
| 10 | What is the capital of France? *(adversarial, out-of-domain)* | none — should decline | French Protectorate of Cambodia, Post Angkor Period (×2), all low scores (0.30–0.35) | N/A (no good source exists) | **Originally no — hallucinated citation** (see below). **Fixed** — now correctly declines before calling the LLM at all. |

**Original score: 8/10 fully correct and grounded, 1/10 partially grounded (tangential citation), 1/10 hallucinated citation on an out-of-domain query.**

**Update — the hallucination in query 10 has been fixed** (`rag/generate.py`): `llm_answer()` now checks the top retrieved chunk's similarity score against a `_MIN_RELEVANCE_SCORE = 0.45` floor *before* calling the LLM at all. That threshold was picked directly from this table's score distribution — every in-domain query above scored 0.59–0.76 top-hit similarity, while the out-of-domain query topped out at 0.35, leaving a clean gap to set the floor in. Below the floor, `llm_answer()` returns a fixed "I don't have information about this in the indexed documents" message naming the closest (rejected) match and its score — no API call, so no chance of a fabricated citation. As a second layer, the prompt itself was also strengthened to explicitly instruct the model to decline rather than answer from outside knowledge, and to never cite a source that doesn't actually support the claim — this covers borderline cases that pass the score floor but still don't fully answer the question.

Re-verified live: query 10 now returns *"I don't have information about this in the indexed documents. The closest match ('French Protectorate Of Cambodia') only scored 0.35 similarity, below the relevance threshold, so I'm not going to guess."* — no hallucination, and no LLM call was made. Re-checked the lowest-scoring **legitimate** query in this table (Funan, 0.589–0.615) to confirm the new floor doesn't false-positive on real matches — it still answers normally, well clear of the 0.45 cutoff.

**Caveat:** this is an empirical threshold tuned to this corpus and embedding model, not a principled constant — a future corpus with more homogeneous or more diverse topics could shift the score distribution and need retuning.

## What worked

- **Embeddings retrieval is strong on in-domain queries.** All 9 in-domain queries retrieved the correct document in the top-3, usually with all 3 slots from the same (correct) document at similarity scores of 0.6–0.75. This is a large improvement over the TF-IDF baseline, which would have missed paraphrased queries with no literal keyword overlap (e.g. "What was the Khmer Empire known for?" has no word in common with the phrase "hydraulic cities" or "Angkor" that answers it).
- **Multi-document synthesis works.** Query 4 (1997 coup) and query 8 (UNTAC) both correctly pulled from and cited multiple distinct source documents, not just one — confirming `top_k > 1` is pulling its weight rather than the model just repeating the single best chunk.
- **The model abstains instead of fabricating causal claims it doesn't have.** Query 9 is the best example: rather than inventing "the civil war was caused by X," the model explicitly said the sources don't state a cause and described only what they do say. This is exactly the desired behavior for a grounded system.
- **Conversation history (Step 6) holds up under evaluation-style querying too** — verified separately (see chat log): a follow-up with no named subject ("When did he die?" after asking about Sihanouk) correctly resolved via the multi-turn history and cited the right source, rather than failing or hallucinating.

## What didn't work

**1. Hallucinated citation on the out-of-domain adversarial query (query 10) — the most important finding. `FIXED`, see the update note above the table.** Asking "What is the capital of France?" — a question with no answer anywhere in the Cambodia-history corpus — should have produced a refusal ("not covered by these sources"). Instead the model answered correctly from its own general knowledge ("Paris") but **cited an unrelated retrieved chunk (*Post Angkor Period*) as if it supported the claim**. This is precisely the failure mode Step 5 warned about ("verify citations actually appear and are correct, not hallucinated") and it reproduced even though the prompt already said "Answer the question using ONLY the sources below" — the instruction alone wasn't sufficient; the model still reached for parametric knowledge when the retrieved context was irrelevant, and dressed that answer up with a plausible-looking (but false) citation instead of admitting the sources didn't cover it. Fixed with a hard similarity-score floor (`_MIN_RELEVANCE_SCORE = 0.45` in `rag/generate.py`) that refuses before ever calling the LLM, plus a strengthened prompt as a second line of defense for borderline cases.

**2. Leaked Wikipedia markup produces garbage section labels in one corpus file.** `data/History-of-Cambodia-1993-E2-80-93prese.txt` contains un-stripped `[edit&action=edit&section=N "Edit section: ...")]` link fragments — leftover MediaWiki edit-links that `clean_text()`'s `_BRACKET_NOTE_RE` regex (`rag/ingest.py`) doesn't catch because it only matches bracketed content up to 60 characters, and these fragments run longer. The result: one retrieved chunk's `section` tag in query 8 literally read `[edit&action=edit&section=2 "Edit section: Political aspects and elections")]` instead of a clean header. This doesn't corrupt the chunk *text* the LLM reads (only the section metadata used for citation display), so it didn't cause a wrong answer here, but it's a real gap in the cleaning regex — worth widening `_BRACKET_NOTE_RE`'s length cap or adding a dedicated `[edit...]` pattern if this file's citations matter for the final write-up.

**3. Retrieval sometimes surfaces a topically-relevant but rhetorically-weak chunk.** Query 1 ("Who led the Khmer Rouge?") retrieved a passage that *mentions* Pol Pot, Ieng Sary, and Khieu Samphan in a quoted aside ("Khieu Samphan and Son Sen later boasted...") rather than a clean, declarative leadership statement — even though *Democratic Kampuchea* (the more obviously correct source) was also retrieved. The LLM still produced a correct answer (it read all 3 chunks, not just the top one), but the citation it chose to name was the weaker source. This is a byproduct of paragraph-level chunking: a paragraph built around an anecdote can outscore a more declarative paragraph if the anecdote happens to use more of the query's vocabulary. Not a bug, but a real edge case of the chunking strategy defended in Step 2.

**4. The corpus has redundant near-duplicate documents for the same topic.** `United Nations Administered Cambodia.txt` and `United Nations Transitional Authority.txt` are two separate Wikipedia articles about the same UNTAC period, both scraped as full documents. Query 8 retrieved from both, which is why the resulting answer reads as somewhat repetitive/choppy — the model is stitching together two overlapping accounts instead of one clean one. Worth deduplicating or merging these two source files in a future revision.

## Embedding choice tradeoff (for the write-up)

TF-IDF (the starter's original baseline) matches on literal token overlap, so it would have failed queries like #6 ("What was the Khmer Empire known for?") whose answer vocabulary ("hydraulic cities," "Angkor") doesn't overlap with the query at all. Swapping to `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings — semantic vectors instead of bag-of-words — is what makes 9/10 queries above retrieve correctly despite little-to-no shared vocabulary between question and answer passage. The cost is real (a ~80MB model download, encoding latency per query — see the latency display added in Step 6), but for this corpus size (1,106 chunks) that cost stays under ~50ms per query, well within interactive UI budget.

## Practical note: Gemini free-tier quota

The free tier caps out at **20 requests/day per project per model**. Running this evaluation (10 queries, plus retries) used a meaningful chunk of that budget in one sitting, and a follow-up verification run hit a `429 RESOURCE_EXHAUSTED` error before completing. The quota is scoped to the Google Cloud **project**, not the individual API key — swapping to a new key generated under the same project does not reset it; a genuinely new project (or a fresh day) is required. Worth knowing before a live demo or grading session: batch your testing, don't re-run the full eval suite repeatedly the same day, and if you hit the quota, `extractive` mode still works with no API calls at all as a fallback to show retrieval quality.

## Update — model swap (`gemini-2.5-flash` → `gemini-flash-latest`)

After this evaluation was run, `gemini-2.5-flash` (the model used for all 10 queries above and for the hallucination fix above) started returning `404 NOT_FOUND: This model ... is no longer available to new users` when tested against a freshly-created Google Cloud project — Google had sunset it for new projects, even though it still appeared in `client.models.list()`. Switched `_MODEL` in `rag/generate.py` to `gemini-flash-latest`, an alias Google maintains to always point at their current default flash model, specifically to avoid pinning to a dated snapshot that can be retired mid-project again. Re-verified live against the actual corpus after the swap — retrieval and generation both work unchanged; the results and analysis above remain representative of current behavior.

## Update — provider swap (Gemini → DeepSeek)

Switched LLM providers from Gemini to DeepSeek (`deepseek-v4-flash`, called via the OpenAI-compatible API at `https://api.deepseek.com`). All results and analysis in this document were originally produced against Gemini; the retrieval side (embeddings, chunking, similarity gate) is provider-independent and unaffected. Re-verified live against DeepSeek after the swap:

- **In-domain query** ("Who was Norodom Sihanouk?") — correctly answered and cited, same as the original Gemini result.
- **Out-of-domain query** ("What is the capital of France?") — still correctly refused via the `_MIN_RELEVANCE_SCORE` similarity gate, which runs before any LLM call and is therefore identical regardless of which provider is behind it.

The prompt, conversation-history handling, and relevance-floor logic in `rag/generate.py` are unchanged in behavior — only the API client and message format (Gemini's `contents`/`parts` shape → OpenAI-style `messages` list) changed to match DeepSeek's API.
