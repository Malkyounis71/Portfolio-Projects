import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Download the NLTK resources for tokenization and stop words
nltk.download('punkt')
nltk.download('stopwords')


def load_data(file_path):
    """
    Load the dataset from the CSV file and return a pandas dataframe.
    """
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """
    Clean the dataset by removing duplicates and missing values.
    """
    # Remove duplicate rows from the dataframe
    df.drop_duplicates(inplace=True)
    # Remove rows with missing values
    df.dropna(inplace=True)


def preprocess_text(text):
    """
    Preprocess the text data by tokenizing, lowercasing, removing stop words, and stemming the words.
    """
    # Tokenize the text into individual words
    tokens = nltk.word_tokenize(text.lower())
    # Remove stop words (common words such as "the", "and", etc.)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Perform stemming (reduce words to their root form)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Join the stemmed words back into a single text string
    return ' '.join(tokens)


def extract_features(df):
    """
    Extract the relevant features from the dataset and combine them into a single text document.
    """
    # Combine the relevant features (Course Name, University, Skills, and Course Description) into a single text document
    df['text_document'] = df['Course Name'] + ' ' + df['University'] + ' ' + \
        df['Skills'] + ' ' + df['Course Description'].apply(preprocess_text)


def vectorize_text(X):
    """
    Convert the text documents to a numerical representation using TF-IDF vectorization.
    """
    # Create a TF-IDF vectorizer object
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer to the text documents and transform them into a matrix of TF-IDF weights
    X = vectorizer.fit_transform(X)
    return X, vectorizer


def calculate_similarity(user_input, X, vectorizer):
    """
    Calculate the cosine similarity between the user's input and each course in the dataset.
    """
    # Transform the user's input into a matrix of TF-IDF weights using the same vectorizer object as before
    user_input_vector = vectorizer.transform([user_input])
    # Calculate the cosine similarity between the user's input and each course in the dataset
    similarity_scores = cosine_similarity(user_input_vector, X)
    return similarity_scores


def get_top_courses(similarity_scores, df, num_courses=5):
    """
    Rank the courses based on their similarity scores and return the top recommended courses.
    """
    # Get the indices of the top recommended courses based on their similarity scores
    top_indices = np.argsort(similarity_scores, axis=1)[
        :, -num_courses:].flatten()[::-1]
    # Get the top recommended courses based on their indices
    top_courses = df.iloc[top_indices]
    return top_courses


def recommend_courses(user_interests, df):
    """
    Given a list of user interests, preprocess the text data, vectorize the text documents,
    calculate the similarity scores, and return the top recommended courses.
    """
    # Preprocess the user's interests by tokenizing, lowercasing, removing stop words, and stemming the words
    user_input = ' '.join([preprocess_text(interest)
                          for interest in user_interests])

    # Combine the relevant features (Course Name, University, Skills, and Course Description) into a single text document
    extract_features(df)

    # Vectorize the text documents using TF-IDF vectorization
    X, vectorizer = vectorize_text(df['text_document'])

    # Calculate the cosine similarity between the user's input and each course in the dataset
    similarity_scores = calculate_similarity(user_input, X, vectorizer)

    # Get the top recommended courses based on their similarity scores
    top_courses = get_top_courses(similarity_scores, df)

    return top_courses


def find_similar_courses(course_name , df):
    """
    Given a course name, find other courses that are similar based on the course name and other features.
    """
    # Get the row of the course with the given name
    course_row = df[df['Course Name'] == course_name].iloc[0]

    # Define the user's interests as the course name and other relevant features
    user_interests = [course_row['Course Name'],
                      course_row['Skills'], course_row['Course Description']]

    # Get the top recommended courses based on the user's interests
    recommended_courses = recommend_courses(user_interests, df)

    return recommended_courses

if __name__ == '__main__':
    load_data()
    clean_data()
    preprocess_text()
    extract_features()
    vectorize_text()
    calculate_similarity()
    find_similar_courses()

# # Eample
# # Load the dataset
# df = load_data('Coursera.csv')
# # Clean the dataset
# clean_data(df)
# # Define the user's interests
# user_interests = ['C++', 'data science', 'machine learning']
# # Get the top recommended courses based on the user's interests
# recommended_courses = recommend_courses(user_interests, df)
# # Print the top recommended courses
# recommended_courses[['Course Name', 'University']]

# # Load the dataset
# df = load_data('Coursera.csv')
# # Clean the dataset
# clean_data(df)
# # Find similar courses to "Data Science and Machine Learning Bootcamp"
# similar_courses = find_similar_courses("Java Decision Programming", df)

# # Print the top recommended courses
# similar_courses[['Course Name', 'University']][1:]