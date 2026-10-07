from data_tools import load_data
from agent_executor import execute_analysis
from agent_reasoner import (
    build_reasoning_context,
    generate_insight,
)


DATASET_PATH = "dataset/amazon.csv"


def run_agent(question, df):
    """
    Run the complete AI Data Analyst Agent pipeline.
    """

    execution_result = execute_analysis(
        question,
        df,
    )

    reasoning_context = build_reasoning_context(
        execution_result
    )

    insight = generate_insight(
        reasoning_context
    )

    return {
        "question": question,
        "intent": execution_result["intent"],
        "tool": execution_result["tool"],
        "plan": execution_result["plan"],
        "result": execution_result["result"],
        "answer": insight,
    }


def main():
    """Run the interactive AI Data Analyst Agent."""

    df = load_data(DATASET_PATH)

    print("=" * 80)
    print("AI DATA ANALYST AGENT")
    print("=" * 80)
    print("Ask a question about the Amazon product dataset.")
    print("Type 'exit' to stop.")

    while True:
        print()

        question = input("Your question: ").strip()

        if question.lower() == "exit":
            print("\nAgent stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        response = run_agent(
            question,
            df,
        )

        print("\n" + "-" * 80)
        print("INTENT:")
        print(response["intent"])

        print("\nTOOL:")
        print(response["tool"])

        print("\nANALYSIS PLAN:")
        for step in response["plan"]:
            print("-", step)

        print("\nFINAL ANSWER:")
        print(response["answer"])

        print("-" * 80)


if __name__ == "__main__":
    main()