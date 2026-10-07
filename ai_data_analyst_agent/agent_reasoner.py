import os

from langchain_google_genai import ChatGoogleGenerativeAI


def build_reasoning_context(execution_result):
    """
    Build structured context for the LLM.
    """

    if execution_result["result"] is None:
        return {
            "question": execution_result["question"],
            "intent": execution_result["intent"],
            "plan": execution_result["plan"],
            "tool": execution_result["tool"],
            "data_available": False,
            "results": None,
        }

    return {
        "question": execution_result["question"],
        "intent": execution_result["intent"],
        "plan": execution_result["plan"],
        "tool": execution_result["tool"],
        "data_available": True,
        "results": execution_result["result"],
    }


def build_reasoning_prompt(reasoning_context):
    """
    Build the prompt sent to the LLM.
    """

    if not reasoning_context["data_available"]:
        return f"""
You are an AI Data Analyst Agent.

User question:
{reasoning_context["question"]}

The available dataset and analysis tools cannot currently
answer this question.

Explain this limitation clearly.
Do not invent data or provide unsupported analysis.
""".strip()

    return f"""
You are an AI Data Analyst Agent.

Your task is to explain data-analysis results using only
the calculated results provided below.

User question:
{reasoning_context["question"]}

Analysis intent:
{reasoning_context["intent"]}

Analysis plan:
{reasoning_context["plan"]}

Analysis tool:
{reasoning_context["tool"]}

Calculated results:
{reasoning_context["results"]}

Rules:
1. Use only the calculated results provided.
2. Do not invent values.
3. Do not perform calculations that are not supported by the results.
4. Clearly distinguish observations from assumptions.
5. Answer the user's question directly.
6. Mention important limitations when relevant.
""".strip()


def create_llm():
    """
    Create the Gemini model used for interpretation.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    )

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
    )


def generate_insight(reasoning_context):
    """
    Generate a natural-language explanation from
    deterministic analysis results.
    """

    prompt = build_reasoning_prompt(
        reasoning_context
    )

    llm = create_llm()

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        return response.content

    if isinstance(response.content, list):
        text_parts = []

        for block in response.content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)

        return "\n".join(text_parts).strip()

    return str(response.content)