# app.py
import streamlit as st
from src.utils import load_state, save_state, run_idea_generator
from src.syntext import analyze_text
from src.syntone_text import analyze_text_emotion
from src.symind import cluster_keypoints

st.set_page_config(page_title="InsightSyn MVP", layout="wide")
st.title("InsightSyn MVP 💡")

# --- Load State ---
state = load_state(default_state={"analysis_history": [], "brainstorm_history": []})

# --- Sidebar Navigation ---
tool = st.sidebar.radio("Choose a Tool", ["Cognitive Analysis", "Idea Brainstorm"])

# --- Cognitive Analysis Tool ---
if tool == "Cognitive Analysis":
    st.header("Cognitive Analysis")
    st.write("Enter text to extract key insights, summary, emotion, and themes.")
    
    text_input = st.text_area("Enter text for analysis:", height=200, key="analysis_input")
    
    if st.button("Analyze"):
        if text_input.strip():
            with st.spinner("Running analysis..."):
                analysis = analyze_text(text_input)
                emotion = analyze_text_emotion(text_input)
                # Cluster the key entities and the summary to find themes
                items_to_cluster = analysis.get('entities', []) + [analysis.get('summary', '')]
                clusters = cluster_keypoints(items_to_cluster, num_clusters=2)
                
                # Save result
                result = {
                    "input": text_input,
                    "analysis": analysis,
                    "emotion": emotion,
                    "themes": clusters
                }
                state['analysis_history'].insert(0, result)
                save_state(state)
        else:
            st.warning("Please enter some text.")

    # Display latest analysis
    if state['analysis_history']:
        latest = state['analysis_history'][0]
        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Primary Emotion", latest['emotion']['primary'].capitalize())
            st.write("**Summary:**")
            st.info(latest['analysis']['summary'])
        with col2:
            st.metric("Sentiment", latest['analysis']['sentiment']['label'].capitalize())
            st.write("**Key Entities:**")
            st.info(", ".join(latest['analysis']['entities']))
        
        st.write("**Thematic Clusters:**")
        st.json(latest['themes'])

# --- Idea Brainstorm Tool ---
elif tool == "Idea Brainstorm":
    st.header("Idea Brainstorm")
    st.write("Use the main LLM to expand on your ideas.")
    
    idea_input = st.text_input("Enter your starting idea:", key="idea_input")
    
    if st.button("Generate"):
        if idea_input.strip():
            with st.spinner("Generating..."):
                response = run_idea_generator(idea_input)
                
                # Save result
                result = {"input": idea_input, "response": response}
                state['brainstorm_history'].insert(0, result)
                save_state(state)
        else:
            st.warning("Please enter an idea.")
    
    # Display brainstorm history
    if state['brainstorm_history']:
        st.subheader("Brainstorm History")
        for item in state['brainstorm_history']:
            with st.expander(f"Idea: {item['input'][:50]}..."):
                st.markdown(item['response'])