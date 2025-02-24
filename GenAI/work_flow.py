from langgraph.graph import START, END, StateGraph
from typing import TypedDict, List, Literal
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
import os
from sentence_transformers import CrossEncoder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from document_loader import dataLoading


class agents:
    def __init__(self, retrieval: dataLoading):

        self.llm = ChatGroq(api_key=os.getenv("Groq_API_KEY"),
                            model="llama-3.1-8b-instant")

        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.get_retrieval = retrieval

    def workflows(self):
        try:
            class Agent(TypedDict):
                messages: List[str]
                decision: Literal["YES", "NO"] | None
                expanded_queries: List[str] | None
                retrieved_docs: List[str] | None
                reranked_docs: List[str] | None
                answer: str | None

            workflow = StateGraph(Agent)

            def router(state: Agent):
                """Determine query relevance to SSM On-Prem"""
                query = state["messages"][-1]  # Get last message
                prompt = f"""
                Determine if this query relates to Git cheat code. Respond ONLY with "YES" or "NO".
                Query: {query}
                """
                decision = self.llm.predict(prompt).strip().upper()
                return {"decision": decision}

            def route_decision(state: Agent):
                if state.get("decision") == "YES":
                    return "rewrite_query"
                return "apologize_and_exit"

            def apologize_and_exit(state: Agent):
                """Handle non-relevant queries"""
                return {"messages": ["I'm sorry, I'm not able to find the information you requested."]}

            def rewrite_query(state: Agent):
                """Generate query variations"""
                original_query = state["messages"][-1]
                prompt = f"Rephrase this question in 3 different ways:\n{original_query}"
                variations = self.llm.predict(prompt)
                print(variations)
                expanded_queries = [original_query] + [v.strip() for v in variations.split("\n")[:3]]
                return {"expanded_queries": expanded_queries}

            def retrieve_docs(state: Agent):
                """Retrieves documents using the ensemble retriever."""
                print("--Now retrieving docs--")
                retrieved_docs = []
                for query in state['expanded_queries']:
                    retrieved_docs.extend(self.get_retrieval.get_relevant_documents(query))
                state['retrieved_docs'] = retrieved_docs[:25]
                print("Documents retrieved", len(state["retrieved_docs"]))
                return state

            def rerank_documents(state: Agent):
                """Reranks retrieved documents using a cross-encoder model."""
                query = state['messages'][-1]
                print("User's Query: ", query)
                retrieved_docs = state['retrieved_docs']
                doc_texts = [doc.page_content for doc in retrieved_docs]
                scores = self.reranker.predict([(query, doc) for doc in doc_texts])
                ranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]
                state['reranked_docs'] = ranked_docs[:15]
                print("Re-ranking successful")
                return state

            def generate_answer(state: Agent):
                """Generates an answer from the top reranked documents."""
                query = state['messages'][-1]
                context = " ".join([doc.page_content for doc in state['reranked_docs']])
                prompt = f"Based on the following context only, answer the question: {query}\nContext: {context}"
                answer = self.llm.predict(prompt)
                state['answer'] = answer
                print("Answer Generated")
                return state

            workflow.add_node("router", router)
            workflow.add_node("apologize_and_exit", apologize_and_exit)
            workflow.add_node("rewrite_query", rewrite_query)
            workflow.add_node("retrieve_docs", retrieve_docs)
            workflow.add_node("rerank_documents", rerank_documents)
            workflow.add_node("generate_answer", generate_answer)

            workflow.add_edge(START, "router")

            workflow.add_conditional_edges(
                "router",
                route_decision,
                {
                    "rewrite_query": "rewrite_query",
                    "apologize_and_exit": "apologize_and_exit"
                }
            )

            workflow.add_edge("rewrite_query", "retrieve_docs")
            workflow.add_edge("retrieve_docs", "rerank_documents")
            workflow.add_edge("rerank_documents", "generate_answer")
            workflow.add_edge("apologize_and_exit", END)
            workflow.add_edge("generate_answer", END)

            initial_state = Agent(
                messages=[],
                decision=None,
                expanded_queries=None,
                retrieved_docs=None,
                reranked_docs=None,
                answer=None
            )

            app = workflow.compile()
            '''result = app.invoke(initial_state)
            print(result['answer'])'''
            return app

        except Exception as e:
            print("Error in workflow", e)


if __name__ == "__main__":
    file_path = "D:/Data Science/docs/"
    path = dataLoading(file_path=file_path)
    retrieving_docs = path.Initiating_data_loader_and_embeddings()
    workflow_start = agents(retrieving_docs)
    workflow_start.workflows()