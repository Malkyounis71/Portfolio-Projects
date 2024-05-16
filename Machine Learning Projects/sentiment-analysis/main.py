import streamlit as st
import pandas as pd
import altair as alt
import re
import string
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore")

# Function to clean a review text
def clean_text(text):
    # Remove leading/trailing white spaces
    text = text.strip()
    
    # Remove URLs using a regular expression
    text = re.sub(r'http\S+', '', text)
    
    # Remove one-word reviews (you can adjust the word length as needed)
    if len(text.split()) <= 1:
        return None
    
    # Remove punctuation and symbols
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # You can add more specific cleaning steps as needed
    
    return text

st.title('Sentiment Analysis❤️')
st.divider()
st.header('Positive😄, Neutral🤷‍♂️, Negative😞')

# Upload your dataset
uploaded_file = st.file_uploader("", type=["csv", "xlsx"])

if st.button('Load Default Data'):
    uploaded_file = "D:\My_Projects\Jupyter\Streamlit-Apps\sentiment-analysis\\twitter_review.csv"

st.divider()

st.subheader('The defualt data has :red[2981] rows but I only use :red[50] rows for test and speed')

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file, nrows=50)  # Modify this if your dataset is in a different format
    df.dropna(inplace=True)

    # Select the column containing review content
    review_column = st.selectbox("Select the column containing review content", df.columns)

    @st.cache_resource(show_spinner=False)
    def load_model(review):
        sentiment_task = pipeline("sentiment-analysis")
        return sentiment_task(review)


    with st.spinner("Analyzing sentiment..."):
        # Analyze sentiment for each cleaned review in the selected column
        results = []
        for review in df[review_column]:
            cleaned_review = clean_text(review)
            if cleaned_review is not None:
                results.append(load_model(cleaned_review))

    st.balloons()
    # Calculate the percentage of positive, neutral, and negative sentiments
    num_reviews = len(results)
    num_positive = sum(1 for result in results if result[0]['label'] == 'POSITIVE')
    num_neutral = sum(1 for result in results if result[0]['label'] == 'NEUTRAL')
    num_negative = sum(1 for result in results if result[0]['label'] == 'NEGATIVE')

    pct_positive = (num_positive / num_reviews) * 100
    pct_neutral = (num_neutral / num_reviews) * 100
    pct_negative = (num_negative / num_reviews) * 100


    # Create a DataFrame for the sentiment percentages
    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative'],
        'Percentage': [pct_positive, pct_neutral, pct_negative]
    })

    # Create a horizontal bar chart using Altair
    chart = alt.Chart(sentiment_data).mark_bar().encode(
        x=alt.X('Percentage:Q', axis=alt.Axis(format='~s')), # Format the x-axis as percentages
        y=alt.Y('Sentiment:N', title=None),
        color=alt.Color('Sentiment:N', legend=None),
        tooltip=['Sentiment', 'Percentage']
    ).properties(
        width=500,
        height=150
    )

    st.divider()

    # Create a container for the sentiment analysis results
    st.subheader("Sentiment Analysis Results")

    # Display the chart using the Altair chart object
    st.altair_chart(chart)

    # Display the percentages in a more clear format with custom text colors
    st.markdown(
        f"Percentage of Positive Sentiments: <span style='color:#008000;'>{pct_positive:.2f}%</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"Percentage of Neutral Sentiments: <span style='color:#FFA500;'>{pct_neutral:.2f}%</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"Percentage of Negative Sentiments: <span style='color:#FF0000;'>{pct_negative:.2f}%</span>",
        unsafe_allow_html=True
    )
