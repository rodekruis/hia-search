# Handling users in distress

**Status:** proposal, not implemented yet.

The chatbot's job is to give information from HIA. Some people who write to it will be upset, frightened or grieving, and the chatbot should handle them gracefully. This plan follows the spirit of psychological first aid (PFA), based on the IFRC PS Centre [Guide to Psychological First Aid](https://mhpsshub.org/wp-content/uploads/2019/05/PFA-Guide-low-res.pdf). The chatbot does not deliver PFA. It notices distress, puts safety first, doesn't probe, and links people to help.

Requirement: no separate list of referrals or contacts to maintain. Support contacts live in HIA, like all other content, and are retrieved when needed.

## Why the prompt alone is not enough

- **Distress messages get no contacts.** A message like "I can't cope anymore" will likely skip retrieval, because `query_or_respond` treats it like small talk. `config/direct_answer_prompt.txt` then forbids stating contacts. The user gets either nothing useful or the fixed "I don't have the right information" reply.
- **Custom prompts would drop the rules.** A sheet's `#system-prompt` replaces `config/rag_agent_prompt.txt` entirely. Distress rules written there would be lost on any instance with a custom prompt.
- **The model writes the search query.** For a distressed person, finding an emergency number should not depend on how the model phrases a search.

## Plan

### 1. A wellbeing prompt that is the same for every instance

Add `config/wellbeing_prompt.txt` and a `get_wellbeing_prompt()` loader in `utils/prompt_loader.py`, mirroring `get_language_prompt()`. Append it in both `query_or_respond` and `generate`, after the instance prompt (and after the direct-answer rules in `query_or_respond`) and before the language rules, which stay last.

Proposed content:

```
# Wellbeing
Some people who write to you may be upset, frightened or grieving. These rules add to the
ones above and take precedence over # Answering when they conflict.
- Briefly and warmly acknowledge distress (e.g. "That sounds really hard"). Do not treat
  ordinary frustration as distress, and do not act as a counsellor or therapist.
- Distress is never small talk. When the human shows distress, call `find_support`, plus
  `retrieve` if they also asked a question.
- Safety first. If the human mentions danger, injury, or thoughts of harming themselves or
  others, put their other request aside, calmly ask whether they are safe right now, and give
  the contacts from `find_support` (need: emergency). Encourage them to reach someone nearby.
- If they seem to be a child in distress, describe a child being harmed or left alone, or
  disclose sexual or gender-based violence, give the matching contacts (need: child_protection
  or gbv) and the contact for reaching a person (need: talk_to_staff).
- Keep replies calm and simple: short sentences, one step at a time.
- Don't ask for details of what happened beyond what you need to help, and don't judge how
  they feel.
- A clear, practical answer to their question is also support: answer it well.
- If they seem overwhelmed or need more than information, offer the contact for talking to a
  person (need: talk_to_staff), without insisting. Tell them they can come back anytime.
- Only give contacts that appear in the documents. If none are found, don't invent any:
  acknowledge how they feel, say you don't have a number for that here, and encourage them to
  contact local emergency services or someone they trust. Never reply only with the fixed
  reply for when no documents are provided.
- If asked, or if they seem to expect a person, say that you are an automated assistant.
```

Also add one line to `config/direct_answer_prompt.txt`, under the rules for when to reply without retrieval:

```
Messages showing distress or risk are never small talk: call the `find_support` tool.
```

### 2. A `find_support` tool backed by the HIA index

Add a second tool in `agents/rag_agent.py`. The search queries are fixed in code, so they are tested once and don't drift with how users phrase things. Results come from the same index as `retrieve`, which already holds Offers with phone numbers, opening hours and addresses.

```python
SupportNeed = Literal["emergency", "emotional_support", "talk_to_staff", "child_protection", "gbv"]

SUPPORT_QUERIES: dict[str, str] = {
    "emergency": "emergency phone number ambulance police urgent help",
    "emotional_support": "psychosocial support mental health helpline someone to talk to",
    "talk_to_staff": "contact the Red Cross speak to a staff member helpdesk",
    "child_protection": "child protection unaccompanied children abuse",
    "gbv": "sexual violence domestic violence gender-based violence support",
}
_SUPPORT_K = 3

_FIND_SUPPORT_DESCRIPTION = (
    "Find contacts in the HIA documents for people who need support: emergency services, "
    "emotional support, a staff member to talk to, child protection, or help after sexual or "
    "gender-based violence. Call it whenever the user shows distress or may be at risk, "
    "once per need."
)


@tool(response_format="content_and_artifact", description=_FIND_SUPPORT_DESCRIPTION)
def find_support(need: SupportNeed, config: RunnableConfig) -> tuple[str, List[Document]]:
    vector_store = get_vector_store(config["configurable"]["googleSheetId"])
    docs = []
    for doc, score in vector_store.similarity_search_with_score(SUPPORT_QUERIES[need], k=_SUPPORT_K):
        doc.metadata[dm.SCORE] = score
        doc.metadata["support_need"] = need
        docs.append(doc)
    return format_context(docs), docs
```

Wiring changes:

- **Register the tool.** Use `bind_tools([retrieve, find_support])` in `query_or_respond` and `ToolNode([retrieve, find_support])` in `build_graph`.
- **Handle parallel calls in `generate()`.** It already collects documents from every tool message after the tool-calling AI message. It needs to drop duplicates, since the same offer can come back from both tools.
- **Fix the tracing.** `search_query` should join the queries from all `retrieve` calls, instead of reading only the first tool call. Add `support_needs` (the list of `need` values called this turn) to the `chat-turn` span metadata so these turns can be filtered in Langfuse.
- **Optional: filter to Offers.** Restricting `find_support` to `source == "Offers"` would avoid Q&As crowding out the contacts. This needs `source` to be a filterable field in the index.

The number of model calls per turn stays at two, so the Twilio limits (`LLM_TIMEOUT_S`, 15-second webhook) don't change.

### 3. A coverage check when the index is built

This replaces a separately maintained referral list. At the end of `/create-vector-store` (`routes/data.py`), run each entry in `SUPPORT_QUERIES` against the new index and record the best score per need. Return a warning for any need whose best score is below `SUPPORT_COVERAGE_THRESHOLD`, for example:

```json
{
  "message": "Created vector store index ... with 412 documents.",
  "support_coverage": {"emergency": 0.41, "emotional_support": 0.38, "talk_to_staff": 0.12, "...": "..."},
  "warnings": ["No contact found for 'talk_to_staff': add an Offer explaining how to reach the National Society."]
}
```

The National Society keeps everything in HIA and just sees when something is missing.

- **Calibrate the threshold.** Run the check on a few existing instances first.
- **Small API change.** The endpoint's response changes from a plain string to JSON.
- **Document it.** Add a short README section listing the content every instance should have: an emergency number, emotional or psychosocial support, how to contact the National Society, and child protection and GBV services where they exist.

### 4. Evaluation

- **Offline tests (mocked, like the existing ones):**
  - The wellbeing rules are appended in both model calls, also when a sheet has its own `#system-prompt`.
  - A `find_support` tool call reaches `generate`, and its documents are passed in the context.
  - Parallel `retrieve` and `find_support` calls are merged without duplicates.
  - The coverage check warns when a need scores below the threshold.
- **Behaviour checks (live, against a test sheet):** about 20 multilingual scenarios, scored with the distress judge described under Evaluators in the README:
  - distress together with an information question
  - explicit risk of harm to self or others
  - a child writing, or a child at risk
  - a disclosure of sexual or gender-based violence
  - grief
  - ordinary annoyance ("this bot is useless"), which must not trigger the distress response
  - an instance with no support contacts, where the bot must not invent numbers

Over-triggering is the likeliest failure, so the annoyance cases matter most.

## Out of scope / known gaps

- **There is no human handover.** Nothing in hia-search can pass a conversation to a person, so "hand over" here means giving the contact from HIA. A live handover would need Twilio-side routing or a flag that staff monitor.
- **The fallback message ignores distress.** If the model call times out, the user gets the generic English `FALLBACK_MESSAGE` from `routes/chat.py`. One option is a fallback in code that adds the instance's emergency contact, cached from the coverage check, so a failed turn still shows a number.
- **Mandatory reporting.** The PFA guide states that reporting child abuse is mandatory. A chatbot cannot report, so this depends on the National Society's own processes behind the `talk_to_staff` contact.
