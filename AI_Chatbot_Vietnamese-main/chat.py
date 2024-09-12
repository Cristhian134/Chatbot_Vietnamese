import json
import random
import torch
from model import MyNeuralNet
from nltk_utils import bag_of_words, token

import keyboard 
from chat_using_LogisticRegression import chat_bot_LR
# from chat_using_phobert_finetuned import chat_bot_PhoBERT

from googletrans import Translator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('content.json', 'r', encoding='utf-8') as json_data:
	contents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
num_class = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = MyNeuralNet(input_size, hidden_size, num_class)
model.load_state_dict(model_state)
model.eval()

def chat_bot(sentence):
	sentence = token(sentence)
	X = bag_of_words(sentence, all_words)
	X = X.reshape(1, X.shape[0])
	X = torch.from_numpy(X)
	output = model(X)
	_, predict = torch.max(output, dim=1)
	tag = tags[predict.item()]

	probs = torch.softmax(output, dim=1)
	prob = probs[0][predict.item()]
	if prob.item() > 0.75:
		for content in contents['intents']:
			if tag == content['tag']:
				answer = random.choice(content['responses'])
	else:
		answer = "I don't understand you, please be more clear!"
	return answer

txt = ""

translator = Translator()

def traducir_a_vietnamita(texto):
	return translator.translate(texto, dest='vi').text

def traducir_a_espanol(texto):
	return translator.translate(texto, dest='es').text

# while True:
# 	txt = input("Ingresar texto: ") 
# 	print("chatbot: ", chat_bot(txt))
# 	query_LR = chat_bot_LR(txt)
# 	print("chatbot_LR: ", query_LR)
# 	query_BERT = chat_bot_PhoBERT(txt)
# 	print("chatbot_BERT: ", query_BERT)
# 	if keyboard.is_pressed('0') or txt == str(0):  # Detecta si la tecla '0' es presionada
# 		print("La tecla '0' ha sido presionada.")
# 		break  # Salir del bucle si se presiona '0'

while True:

	# Entrada del usuario en español
	txt = input("Ingresar texto (Español): ") 
	
	# Traducir el texto de entrada al vietnamita
	texto_vietnamita = traducir_a_vietnamita(str(txt))
	print("Texto traducido al vietnamita: ", texto_vietnamita)
	

	# # Chatbot procesa el texto en vietnamita
	# respuesta_chatbot = chat_bot(texto_vietnamita)
	# print("Respuesta del chatbot (Vietnamita): ", respuesta_chatbot)
	# # Traducir la respuesta del chatbot de vietnamita a español
	# respuesta_espanol = traducir_a_espanol(respuesta_chatbot)
	# print("Respuesta del chatbot (Español): ", respuesta_espanol)
	
	# Chatbot con el modelo LR
	query_LR = chat_bot_LR(texto_vietnamita)
	print("Respuesta chatbot_LR (Vietnamita): ", query_LR)
	query_LR_espanol = traducir_a_espanol(query_LR)
	print("Respuesta chatbot_LR (Español): ", query_LR_espanol)
	

	# Chatbot con el modelo PhoBERT
	# query_BERT = chat_bot_PhoBERT(texto_vietnamita)
	# print("Respuesta chatbot_BERT (Vietnamita): ", query_BERT)
	# query_BERT_espanol = traducir_a_espanol(query_BERT)
	# print("Respuesta chatbot_BERT (Español): ", query_BERT_espanol)

	# Detectar si se presiona la tecla '0' para salir
	if keyboard.is_pressed('0') or txt == str(0):  
		print("La tecla '0' ha sido presionada.")
		break  # Salir del bucle si se presiona '0'