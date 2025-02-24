from document_loader import dataLoading
from work_flow import agents

file_path = "D:/Data Science/docs/"
doc_path = dataLoading(file_path=file_path)

if __name__=="__main__":
    file_path = "D:/Data Science/docs/"
    doc_path = dataLoading(file_path=file_path)
    retrieving_docs = doc_path.Initiating_data_loader_and_embeddings()
    workflow_start = agents(retrieving_docs)
    workflow_app = workflow_start.workflows()