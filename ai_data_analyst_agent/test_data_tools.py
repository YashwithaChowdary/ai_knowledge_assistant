from data_tools import (
    load_data,
    top_rated_products,
    highest_discount_products,
    category_product_counts,
    average_rating_by_category,
    most_reviewed_products,
    highly_reviewed_low_rated_products,
)


df = load_data("dataset/amazon.csv")

print("Dataset shape:", df.shape)

print("\nTop-rated products:")
print(top_rated_products(df, 5).to_string(index=False))

print("\nHighest-discount products:")
print(highest_discount_products(df, 5).to_string(index=False))

print("\nTop categories by product count:")
print(category_product_counts(df).head(5).to_string(index=False))

print("\nHighest average-rated categories:")
print(average_rating_by_category(df).head(5).to_string(index=False))

print("\nMost-reviewed products:")
print(most_reviewed_products(df, 5).to_string(index=False))

print("\nHighly reviewed but low-rated products:")
print(
    highly_reviewed_low_rated_products(df).to_string(index=False)
)