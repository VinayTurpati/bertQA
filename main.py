from flask import Flask, render_template, request
import torch
from model import bert_model 
from get_results import answering_question, process, search2
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
run_with_ngrok(app)
model_type = 'bert-large-cased-whole-word-masking-finetuned-squad'
model, tokenizer = bert_model(model_type)

@app.route("/")
@app.route("/home")
def home():
	question = ""
	return render_template('index.html', question = question)

@app.route('/search',methods = ['POST'])
def search():
	if request.method == "POST":
		question = request.form.get('question')
		num_results = 10
		answer = answering_question(model, tokenizer, question, num_results)
		response = answer.capitalize()
		return render_template('search.html',  question = question, response = response)

if __name__ == '__main__':
	# Threaded option to enable multiple instances for multiple user access support
	app.run(debug = True,threaded=True, port=2000)
