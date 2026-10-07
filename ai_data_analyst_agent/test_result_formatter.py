from data_tools import load_data, top_rated_products
from result_formatter import format_analysis_result


df = load_data("dataset/amazon.csv")

result = top_rated_products(df, 5)

formatted = format_analysis_result(result)

print("ROW COUNT:")
print(formatted["row_count"])

print("\nCOLUMNS:")
print(formatted["columns"])

print("\nROWS:")
for row in formatted["rows"]:
    print(row)