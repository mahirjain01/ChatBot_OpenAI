from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import OpenAI
import streamlit as st
import os

def main():
    st.set_page_config(page_title="Leveraging GPT for Question Answering on CSV Documents")
    st.header("Ask anything!!")

    # Allow the user to input the OpenAI API key
    user_input_api_key = st.text_input("Enter your OpenAI API key:")

    if not user_input_api_key:
        st.warning("Please enter your OpenAI API key.")
        return

    # Set the OpenAI API key as an environment variable
    os.environ["OPENAI_API_KEY"] = user_input_api_key

    try:
        openai_instance = OpenAI()
    except Exception as e:
        st.error(f"Error creating OpenAI instance: {e}")
        return

    # Continue with the rest of the code
    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        agent = create_csv_agent(openai_instance, csv_file, verbose=True)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))

if __name__ == "__main__":
    main()
