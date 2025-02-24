from langchain_community.document_loaders.generic import GenericLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever


class dataLoading:
    def __init__(self, file_path):
        self.file_path = file_path

    def data_loader(self):
        print("---inside dir---")
        try:
            loader = GenericLoader.from_filesystem(
                self.file_path,
                glob="*",
                suffixes=[".pdf"]
            )
            load_documents = loader.load()
            print("documents loaded successfully", len(load_documents))
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
            split_documents = text_splitter.split_documents(load_documents)
            print("Document split successfull", len(split_documents))
            return split_documents
        except Exception as e:
            print("Error loading file", e)

    def creating_embeddings(self, splitted_documents):
        print("Creating_embeddings...")
        try:
            huggingface_embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            db = Chroma.from_documents(splitted_documents, huggingface_embeddings)
            print("Embeddings created successfully")
            return db
        except Exception as e:
            print("Error in creating embeddings", e)

    def retrieve(self, embedding):
        try:
            if embedding is None:
                print("no embedding found")
                return None
            # Retrieving documents from vector store
            retriever_mmr = embedding.as_retriever(search_type="mmr", search_kwargs={"k": 60})
            retriever_similarity = embedding.as_retriever(search_type="similarity", search_kwargs={"k": 60})
            # initialize the ensemble retriever with Retrievers
            ensemble_retriever = EnsembleRetriever(
                retrievers=[retriever_similarity, retriever_mmr], weights=[1, 1]
            )
            print("retrieval done successfully")
            return ensemble_retriever
        except Exception as e:
            print("Error in retrieving documents", e)

    def Initiating_data_loader_and_embeddings(self):
        try:
            documents = self.data_loader()
            embeddings_created = self.creating_embeddings(documents)
            print("Embeddings stored in chroma db successfully!!")
            retrieve_documents = self.retrieve(embedding=embeddings_created)
            return retrieve_documents
        except Exception as e:
            print("Initiate data loader and embeddings failed", e)


if __name__ == "__main__":
    file_path = "D:/Data Science/docs/"
    path = dataLoading(file_path=file_path)
    docs_retrieved = path.Initiating_data_loader_and_embeddings()


