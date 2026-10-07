import streamlit as st

from data_tools import load_data
from agent import run_agent


DATASET_PATH = "dataset/amazon.csv"


st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="AI",
    layout="wide",
)


@st.cache_data
def load_dataset():
    return load_data(DATASET_PATH)


def main():
    st.title("AI Data Analyst Agent")

    st.write(
        "Ask natural-language questions about the Amazon "
        "product and review dataset."
    )

    st.info(
        "The dataset contains product, pricing, discount, "
        "rating, and review information. It does not contain "
        "sales dates or sales quantities."
    )

    df = load_dataset()

    st.sidebar.header("Dataset")

    st.sidebar.write(
        f"Rows: {df.shape[0]}"
    )

    st.sidebar.write(
        f"Columns: {df.shape[1]}"
    )

    st.sidebar.write(
        "Source: dataset/amazon.csv"
    )

    question = st.text_input(
        "Ask a data-analysis question",
        placeholder="Example: Which products have the largest discounts?",
    )

    if st.button("Analyze"):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Analyzing the dataset..."):
            response = run_agent(
                question,
                df,
            )

        st.subheader("Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Intent**")
            st.code(response["intent"])

        with col2:
            st.write("**Tool**")
            st.code(
                response["tool"]
                if response["tool"]
                else "No tool selected"
            )

        st.write("**Analysis Plan**")

        for step in response["plan"]:
            st.write(f"- {step}")

        st.subheader("Answer")

        st.write(response["answer"])

        if response["result"] is not None:
            st.subheader("Calculated Results")

            rows = response["result"]["rows"]

            if rows:
                st.dataframe(
                    rows,
                    use_container_width=True,
                )
            else:
                st.info("No matching results were found.")
        else:
            st.info(
                "No deterministic analysis result was available "
                "for this question."
            )


if __name__ == "__main__":
    main()