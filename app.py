import streamlit as st
import os
import json
from dotenv import load_dotenv
from groq import Groq
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

with open("knowledge.txt", "r") as f:
    documents = f.readlines()

embeddings = embed_model.encode(documents)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(np.array(embeddings))


def retrieve_context(query):

    query_embedding = embed_model.encode(
        [query]
    )

    distances, indices = index.search(
        np.array(query_embedding),
        3
    )

    context = ""

    for i in indices[0]:
        context += documents[i]

    return context

st.set_page_config(
    page_title="AI Sports Quiz Generator",
    page_icon="🏆"
)

if "quiz" not in st.session_state:
    st.session_state.quiz = []


st.title("🏆 AI Sports Quiz Generator")
st.write("Generate AI-powered sports quizzes!")


sport = st.selectbox(
    "Select Sport",
    ["Cricket", "Football", "Tennis", "Basketball"]
)

difficulty = st.selectbox(
    "Select Difficulty",
    ["Easy", "Medium", "Hard"]
)


if st.button("Generate Quiz"):

    context = retrieve_context(
        f"{sport} {difficulty} sports quiz"
    )

    prompt = f"""
Use the following sports information:

{context}

Generate exactly 5 {difficulty} level multiple-choice questions about {sport}.

Rules:
- Each question must have exactly 4 options.
- Only one option should be correct.
- Do not repeat questions.
- Return ONLY valid JSON.

Format:
[
  {{
    "question": "Question",
    "options": ["A", "B", "C", "D"],
    "answer": "Correct Option"
  }}
]
"""


    with st.spinner("Generating Quiz..."):

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = response.choices[0].message.content

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()


        try:
            quiz = json.loads(text)

            st.session_state.quiz = quiz
            st.success("AI Quiz Generated Successfully!")

        except json.JSONDecodeError:
            st.error("Failed to generate quiz. Try again.")



if st.session_state.quiz:

    user_answers = {}

    st.subheader(f"{sport} Quiz")

    for i, q in enumerate(st.session_state.quiz, 1):

        st.write(f"### Q{i}. {q['question']}")

        user_answers[i] = st.radio(
            f"Choose answer for Q{i}",
            q["options"],
            key=f"q{i}"
        )


    if st.button("Submit Quiz"):

        score = 0

        for i, q in enumerate(st.session_state.quiz, 1):

            if user_answers[i] == q["answer"]:
                score += 1


        st.success(f"🎉 Your Score: {score}/5")

        st.subheader("Quiz Review")


        for i, q in enumerate(st.session_state.quiz, 1):

            st.write(f"**Q{i}: {q['question']}**")
            st.write(f"Your Answer: {user_answers[i]}")
            st.write(f"Correct Answer: {q['answer']}")

            if user_answers[i] == q["answer"]:
                st.success("✅ Correct")
            else:
                st.error("❌ Wrong")


    if st.button("🔄 Play Again"):

        st.session_state.quiz = []
        st.rerun()