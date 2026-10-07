from data_tools import load_data
from agent_executor import execute_analysis


df = load_data("dataset/amazon.csv")


questions = [
    "Which products have the highest ratings?",
    "Which products have the largest discounts?",
    "Which categories have the most products?",
    "Which categories have the highest average rating?",
    "Which products have the most reviews?",
    "Which products have many reviews but low ratings?",
    "Why did sales decrease last month?",
]


for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)

    result = execute_analysis(question, df)

    print("INTENT:", result["intent"])
    print("TOOL:", result["tool"])

    print("PLAN:")
    for step in result["plan"]:
        print("-", step)

    print("RESULT:")

    if result["result"] is None:
        print("This question is not currently supported.")
    else:
        print("Rows:", result["result"]["row_count"])
        print("Columns:", result["result"]["columns"])

        for row in result["result"]["rows"][:5]:
            print(row)