def create_analysis_plan(question):
    """
    Create a deterministic analysis plan from a user's question.

    The planner maps supported natural-language questions
    to the appropriate analysis tool.
    """

    question = question.lower().strip()

    # Category average rating must be checked before
    # the general "highest rating" condition.
    if (
        "average" in question
        and "rating" in question
        and "categor" in question
    ):
        return {
            "intent": "average_rating_by_category",
            "plan": [
                "Validate the rating and category data.",
                "Convert ratings to numeric values.",
                "Group products by category.",
                "Calculate the average rating for each category.",
                "Rank categories by average rating.",
                "Return the results for interpretation.",
            ],
            "tool": "average_rating_by_category",
        }

    if "highest" in question and "rating" in question:
        return {
            "intent": "find_top_rated_products",
            "plan": [
                "Validate the rating data.",
                "Sort products by rating in descending order.",
                "Select the highest-rated products.",
                "Return the results for interpretation.",
            ],
            "tool": "top_rated_products",
        }

    if "largest" in question and "discount" in question:
        return {
            "intent": "find_highest_discount_products",
            "plan": [
                "Validate the discount data.",
                "Sort products by discount percentage.",
                "Select products with the highest discounts.",
                "Return the results for interpretation.",
            ],
            "tool": "highest_discount_products",
        }

    if "most" in question and "products" in question and "categor" in question:
        return {
            "intent": "count_products_by_category",
            "plan": [
                "Validate the category data.",
                "Group products by category.",
                "Count products in each category.",
                "Rank categories by product count.",
                "Return the results for interpretation.",
            ],
            "tool": "category_product_counts",
        }

    if "most reviewed" in question or "most reviews" in question:
        return {
            "intent": "find_most_reviewed_products",
            "plan": [
                "Validate the rating-count data.",
                "Convert rating counts to numeric values.",
                "Sort products by number of ratings.",
                "Select the most-reviewed products.",
                "Return the results for interpretation.",
            ],
            "tool": "most_reviewed_products",
        }

    if (
        ("many reviews" in question or "highly reviewed" in question)
        and ("low rating" in question or "low rated" in question)
    ):
        return {
            "intent": "find_highly_reviewed_low_rated_products",
            "plan": [
                "Validate rating and rating-count data.",
                "Convert ratings and rating counts to numeric values.",
                "Filter products with many reviews and low ratings.",
                "Rank the matching products.",
                "Return the results for interpretation.",
            ],
            "tool": "highly_reviewed_low_rated_products",
        }

    return {
        "intent": "unsupported_question",
        "plan": [
            "Determine whether the question can be answered using the available dataset and tools.",
            "If not supported, explain that the requested analysis is currently unavailable.",
        ],
        "tool": None,
    }