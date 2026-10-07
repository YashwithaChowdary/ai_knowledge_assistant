from data_tools import load_data
from agent_executor import execute_analysis
from agent_reasoner import (
    build_reasoning_context,
    generate_insight,
)


df = load_data("dataset/amazon.csv")

questions = [
    "Which products have the highest ratings?",
    "Why did sales decrease last month?",
]


for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)

    execution_result = execute_analysis(
        question,
        df,
    )

    reasoning_context = build_reasoning_context(
        execution_result
    )

    print("\nAI INSIGHT:")

    try:
        insight = generate_insight(
            reasoning_context
        )

        print(insight)

    except Exception as error:
        print("ERROR:")
        print(type(error).__name__)
        print(error)