# 🏆 AI Sports Quiz Generator

## Project Description
AI Sports Quiz Generator is an AI-powered application that generates sports-based multiple-choice quizzes using Large Language Models and RAG (Retrieval Augmented Generation).

## Features
- Generate sports quizzes using AI
- Select different sports (Cricket, Football, Tennis, Basketball)
- Choose difficulty level (Easy, Medium, Hard)
- RAG-based information retrieval using FAISS
- Automatic quiz score calculation

## Technologies Used
- Python
- Streamlit
- Groq LLM
- FAISS Vector Database
- Sentence Transformers

## How It Works
1. User selects sport and difficulty.
2. Relevant sports information is retrieved from the FAISS vector database.
3. Groq AI generates quiz questions using the retrieved context.
4. User answers questions and receives a score.

## Installation

Install required libraries:

pip install -r requirements.txt

## Run the Application

streamlit run app.py

## Output
The application generates 5 multiple-choice questions and evaluates the user's score.

## Author
Neha Desireddy