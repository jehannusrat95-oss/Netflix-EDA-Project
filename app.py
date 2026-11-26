import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px                                                                                         
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    return df

df = load_data()
st.subheader("Scatter Plot Analysis")

fig = px.scatter(
    df,
    x="release_year",
    y="duration",
    color="type",
    title="Release Year vs Duration"
)

st.plotly_chart(fig)
# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("netflix_titles.csv")

df = load_data()

st.title("Netflix Movies and TV Shows EDA")

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Dataset Shape")
st.write(df.shape)

st.subheader("Dataset Information")
st.write(df.info())

st.subheader("Statistical Summary")
st.write(df.describe(include="all"))
st.header("Data Cleaning")

st.subheader("Missing Values")
st.write(df.isnull().sum())
st.bar_chart(df.isnull().sum())

st.subheader("Duplicate Rows")
st.write(df.duplicated().sum())

# Remove Duplicates
df = df.drop_duplicates()

# Fill Missing Values
df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Not Rated")

st.subheader("Cleaned Dataset")
st.write(df.isnull().sum())

st.write("Final Dataset Shape:", df.shape)
st.header("Netflix Data Visualization")

# Movies and TV Shows
st.subheader("Movies vs TV Shows")

type_count = df["type"].value_counts()

fig = px.pie(
    values=type_count.values,
    names=type_count.index,
    title="Movies vs TV Shows"
)
st.plotly_chart(fig)

# Top 10 Countries
st.subheader("Top 10 Countries")

country_count = (
    df[df["country"] != "Unknown"]["country"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

fig = px.bar(
    x=country_count.index,
    y=country_count.values,
    title="Top 10 Countries"
)
st.plotly_chart(fig)
st.header("Release Year Analysis")

year_count = df["release_year"].value_counts().sort_index()

fig = px.line(
    x=year_count.index,
    y=year_count.values,
    title="Content Released by Year",
    labels={
        "x": "Release Year",
        "y": "Number of Shows"
    }
)

st.plotly_chart(fig)
st.header("Content Rating Analysis")

rating_count = df["rating"].value_counts()

fig = px.bar(
    x=rating_count.index,
    y=rating_count.values,
    title="Netflix Content Ratings",
    labels={
        "x": "Rating",
        "y": "Number of Titles"
    }
)

st.plotly_chart(fig)
st.header("Movie Duration Analysis")

movies = df[df["type"] == "Movie"].copy()

movies["duration_num"] = (
    movies["duration"]
    .str.extract(r"(\d+)")
    .astype(float)
)

fig = px.histogram(
    movies,
    x="duration_num",
    nbins=20,
    title="Distribution of Movie Duration",
    labels={"duration_num": "Duration (Minutes)"}
)

st.plotly_chart(fig)
st.header("Top 10 Genres Analysis")

genre_count = (
    df["listed_in"]
    .dropna()
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

fig = px.bar(
    x=genre_count.values,
    y=genre_count.index,
    orientation="h",
    title="Top 10 Netflix Genres",
    labels={
        "x": "Number of Titles",
        "y": "Genre"
    }
)

st.plotly_chart(fig)
st.header("Netflix Interactive Filters")

selected_type = st.selectbox(
    "Select Content Type",
    ["All", "Movie", "TV Show"]
)

if selected_type == "All":
    filtered_df = df
else:
    filtered_df = df[df["type"] == selected_type]

st.subheader("Filtered Netflix Data")
st.dataframe(filtered_df)

st.write("Total Titles:", len(filtered_df))
st.header("Netflix Key Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Titles", len(df))

with col2:
    st.metric("Total Movies", len(df[df["type"] == "Movie"]))

with col3:
    st.metric("Total TV Shows", len(df[df["type"] == "TV Show"]))
    st.header("Country Wise Netflix Content")

country_data = (
    df[df["country"] != "Unknown"]["country"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

fig = px.bar(
    x=country_data.index,
    y=country_data.values,
    title="Top 10 Countries by Netflix Content",
    labels={
        "x": "Country",
        "y": "Number of Titles"
    }
)

st.plotly_chart(fig)
st.header("Project Findings")

st.write("""
1. Netflix contains both Movies and TV Shows.
2. Movies are more common than TV Shows.
3. Netflix content comes from many countries.
4. TV-MA and TV-14 are common content ratings.
5. Netflix has content from different genres.
6. Content release increased over the years.
""")

st.success("Netflix EDA Project Completed!")
