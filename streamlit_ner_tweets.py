import streamlit as st
from openai import OpenAI
import json


st.title('Reading Airline Names from Tweets: Demo')
st.header('Enter a Tweet below and watch as AI extracts the airline name from the tweet!', divider='rainbow')

# OpenAI API key
key = st.text_area("Enter your OpenAI API key here:")
if key:
    client = OpenAI(api_key=key)

# Text input area
user_input = st.text_area("Enter your Tweet here:", height=200)

def get_openai_ner_response(client: OpenAI, tweet_text: str, model="gpt-3.5-turbo") -> str:

    # system prompt
    # This system prompt uses roleplay to embody the task of an expert NER model
    system_prompt = """You are an expert NER model. Your task is to read a tweet and identify any airline
                    companies mentioned. Return a JSON object with a list of identified airline names.
                    If no airline names are found, return an empty list. Do not include other types of entities.
                    Consider common abbreviations (e.g., 'AA' for American Airlines) as well, if relevant."""

    # user prompt
    # This user prompt uses several prompt engineering techniques to guide the model towards the desired output
    # It provides examples (few shot learning), instructions to avoid common pitfalls, and a specific format to follow (JSON output)
    user_prompt = f"""
    Below are examples of tweets and the corresponding JSON output.

    Example 1:
    Tweet: "Had a terrible experience with United yesterday!"
    Output: {{"airlines": ["United Airlines"]}}

    Example 2:
    Tweet: "I just booked a flight on Southwest, I hope it goes well."
    Output: {{"airlines": ["Southwest Airlines"]}}

    Example 3:
    Tweet: "AA lost my luggage again."
    Output: {{"airlines": ["American Airlines"]}}

    Example 4:
    Tweet: "No airline can beat Emirates in terms of luxury."
    Output: {{"airlines": ["Emirates"]}}

    ---
    Now, follow the exact same format and process the tweet below.
    Tweet: "{tweet_text}"

    Remember:
    1. Only list airline names if they are explicitly or implicitly mentioned.
    2. Consider common abbreviations or short forms.
    3. Return the output as a JSON object with the key 'airlines' pointing to a list of strings.
    4. If no airline name is mentioned, return {{"airlines": []}}.
    Remember to include the full airline name, not just the abbreviation.
    Some airlines may be misspelled in the tweet, but you should still include them in the output with the correct name."""


    # Call the OpenAI API
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,  # For more deterministic output
    )

    # print(completion.choices[0].message) #debug

    return json.loads(completion.choices[0].message.content)

# Button to submit the request
if st.button("Extract"):
    if user_input.strip():
        with st.spinner("Extracting..."):
            result = get_openai_ner_response(client, user_input)
        st.subheader("Extraction Result:")
        st.write(result)
    else:
        st.warning("Please enter some text first!")