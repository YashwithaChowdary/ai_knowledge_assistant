from data_tools import (
    load_data,
    top_rated_products,
    highest_discount_products,
    category_product_counts,
    average_rating_by_category,
    most_reviewed_products,
    highly_reviewed_low_rated_products,
)

from agent_planner import create_analysis_plan
from result_formatter import format_analysis_result


def execute_analysis(question, df):
    """
    Create an analysis plan, execute the selected
    deterministic data-analysis tool, and format
    the result for the AI layer.
    """

    analysis = create_analysis_plan(question)

    tool_name = analysis["tool"]

    if tool_name is None:
        return {
            "question": question,
            "intent": analysis["intent"],
            "plan": analysis["plan"],
            "tool": None,
            "result": None,
        }

    if tool_name == "top_rated_products":
        result = top_rated_products(df, 10)

    elif tool_name == "highest_discount_products":
        result = highest_discount_products(df, 10)

    elif tool_name == "category_product_counts":
        result = category_product_counts(df).head(10)

    elif tool_name == "average_rating_by_category":
        result = average_rating_by_category(df).head(10)

    elif tool_name == "most_reviewed_products":
        result = most_reviewed_products(df, 10)

    elif tool_name == "highly_reviewed_low_rated_products":
        result = highly_reviewed_low_rated_products(df, 10)

    else:
        raise ValueError(f"Unknown analysis tool: {tool_name}")

    formatted_result = format_analysis_result(result)

    return {
        "question": question,
        "intent": analysis["intent"],
        "plan": analysis["plan"],
        "tool": tool_name,
        "result": formatted_result,
    }


def main():
    """Run a simple command-line test."""

    df = load_data("dataset/amazon.csv")

    question = "Which products have the highest ratings?"

    result = execute_analysis(question, df)

    print("\nQUESTION:")
    print(result["question"])

    print("\nINTENT:")
    print(result["intent"])

    print("\nTOOL:")
    print(result["tool"])

    print("\nANALYSIS PLAN:")
    for step in result["plan"]:
        print("-", step)

    print("\nRESULT:")
    if result["result"] is None:
        print("This question is not currently supported.")
    else:
        print("Rows:", result["result"]["row_count"])
        print("Columns:", result["result"]["columns"])

        for row in result["result"]["rows"]:
            print(row)


if __name__ == "__main__":
    main()