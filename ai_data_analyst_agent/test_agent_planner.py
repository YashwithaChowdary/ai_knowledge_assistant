from agent_planner import create_analysis_plan


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
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    result = create_analysis_plan(question)

    print("INTENT:", result["intent"])
    print("TOOL:", result["tool"])
    print("PLAN:")

    for step in result["plan"]:
        print("-", step)