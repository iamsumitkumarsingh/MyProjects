from flask import Flask, request, render_template
from main import doc_path, agents

retrieving_docs = doc_path.Initiating_data_loader_and_embeddings()
workflow_start = agents(retrieving_docs)
workflow_app = workflow_start.workflows()

flask_app = Flask(__name__)

@flask_app.route('/',methods=['GET','POST'])
def index():
    if request.method =='POST':
        user_input = request.form['question']
        initial_state = {
            "messages":[user_input],
            "decision":None,
            "expanded_queries":None,
            "retrieved_docs":None,
            "reranked_docs":None,
            "answer":None
        }
        result = workflow_app.invoke(initial_state)
        return render_template('index.html',answer = result['answer'])
    return render_template('index.html')

if __name__=="__main__":
    flask_app.run(host = "0.0.0.0", port = 8000 ,debug=True)