# ==========================================================
# AGENT 0: QUERY AGENT
# ==========================================================

def query_agent(state):
    query = state["symptoms"]
    response_text = "I'm sorry, I cannot answer that query from the available information."

    # Attempt to answer 'what is my last symptom?'
    if "what is my last symptom?" in query.lower() or "what is my last symptoms?" in query.lower():
        if STM_MEMORY:
            last_triage = STM_MEMORY[-1]
            if "symptoms" in last_triage:
                response_text = f"Your last recorded symptom was: {last_triage['symptoms']}"
            else:
                response_text = "No specific symptoms found in the last STM entry."
        else:
            response_text = "No previous symptoms found in short-term memory."

    # Attempt to answer 'what was the risk score?'
    elif "what was the risk score?" in query.lower():
        if STM_MEMORY:
            last_triage = STM_MEMORY[-1]
            if "risk_score" in last_triage:
                response_text = f"The risk score for the last triage was: {last_triage['risk_score']}"
            else:
                response_text = "No risk score found in the last STM entry."
        else:
            response_text = "No previous triage data found in short-term memory."

    # Attempt to answer 'explain the reasoning' or similar
    elif "explain the reasoning" in query.lower() or "tell me more about" in query.lower():
        if STM_MEMORY:
            last_triage = STM_MEMORY[-1]
            if "final_output" in last_triage and "reasoning" in last_triage["final_output"]:
                response_text = f"Here is the reasoning from the last triage: {last_triage['final_output']['reasoning']}"
            else:
                response_text = "No detailed reasoning found in the last STM entry."
        else:
            response_text = "No previous triage data found to explain reasoning."

    # General LLM query if specific patterns not matched but it's clearly a query
    elif query.strip() != "": # If it's not empty, try a general LLM response
        try:
            # Construct a prompt for the LLM to answer the query based on available STM/LTM
            stm_context = f"Recent patient symptoms: {STM_MEMORY[-1]['symptoms'] if STM_MEMORY else 'None'}"
            ltm_context = f"Historical cases: {state['ltm_cases']}"
            llm_query_prompt = f"Given the following context: {stm_context}. {ltm_context}. User asks: {query}. Provide a concise answer."

            llm_response = llm.invoke([
                SystemMessage(content="You are a helpful assistant providing information based on patient data."),
                HumanMessage(content=llm_query_prompt)
            ])
            response_text = llm_response.content
        except Exception as e:
            print(f"Error in general LLM query: {e}")
            response_text = "I encountered an error trying to process your general query."

    state["query_response"] = response_text
    print("✓ Query Agent Completed")
    return state
