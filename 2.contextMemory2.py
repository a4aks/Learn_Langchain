import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.runnables.history import RunnableWithMessageHistory

#===================
#1. STORAGE CONFIGURATION
#===============================

USER_FILE = "users.json"
HISTORY_FOLDER = "chat_history"

os.makedirs(HISTORY_FOLDER, exist_ok = True)

#====================
#2. Load USERS
#======================

def load_users():
	if not os.path.exists(USER_FILE):
		return {}
	with open(USER_FILE, "r") as file:
		return json.load(file)

#====================
#3. SAVE USERS
#===================

def save_users(users):
	with open(USER_FILE, "w") as file:
		json.dump(users, file,indent=2)

#======================
#4. USER SIGN UP
#=======================

def signup(users):
	print("\n-- SIGNUP---")
	name = input("Enter Name:").strip()
	username = input("Enter Username: ").strip().lower()
	password = input("Enter Password:")

	if not name or not username or not password:
		print("All fields are required")
		return

	if not username.replace("_","").isalnum():
		print("Use only letters, numbers and underscores")
		return

	if username in users:
		print("Username already exists")
	users[username] = {
		"name": name,
		"password": password
		}
	save_users(users)
	print("Signup successful")

#=======================
#5. USER LOGIN
#========================

def login(users):
	print("\n---Login---")
	username = input("Enter Username:").strip().lower()
	password = input("Enter Password:")

	if username not in users:
		print("Invalid Username")
		return None

	if users[username]["password"] == password:
		print("Login Successful!")
		print("Welcome", users[username]["name"])
		return username

	print("Incorrect Password")
	return None

#===============
#6. FILE CHAT MESSAGE HISTORY
#===============

class FileChatMessageHistory(BaseChatMessageHistory):
	
	def __init__(self, session_id):
		self.file_path = os.path.join(HISTORY_FOLDER, session_id+ ".json")

	# READ CHAT HISTORY
	@property
	def messages(self):
		if not os.path.exists(self.file_path):
			return []
		with open(self.file_path, "r") as file:
			data = json.load(file)

		return messages_from_dict(data)

	#SAVE CHAT HISTORY
	def add_messages(self, messages):
		all_messages = self.messages + messages

		data = [message_to_dict(message) for message in all_messages]

		with open(self.file_path, "w") as file:
			json.dump(data, file, indent=2)

	# CLEAR CHAT HISTORY
	def clear(self):
		with open(self.path, "w") as file:
			json.dump([], file)

#==========================
#7. CREATE OPENAI MODEL
#==========================

llm = ChatOpenAI(model = "gpt-5.6-luna", api_key=os.getenv("OPENAI_API_KEY"))


#===========================
#8. CREATE PROMPT
#==========================

prompt = ChatPromptTemplate.from_messages([
	(
	"system", 
	"You are a helpful AI assistant."
	"Answer in simple English. "
	"The users name is {name}."
	),
	MessagesPlaceholder("history"),
	("human", "{question}")
])

#========================
#9. CREATE CHAIN
#===========================

chain = prompt | llm

#================
#10. GET SESSION HISTORY
#==========================

def get_session_history(session_id):
	return FileChatMessageHistory(session_id)

#=============
#11. CREATE CHATBOT WITH HISTORY
#==========================

chatbot = RunnableWithMessageHistory(
		chain,
		get_session_history,
		input_messages_key="question",
		history_messages_key="history"
	  )

#=====================
#12. LOAD REGISTERED USERS
#=====================

users = load_users()
current_user = None

#=========================
#13. START APPLICATION
#===========================

print("\n MULTI-USER AI CHATBOT")

while True:
	#USER NOT LOGGED IN

	if current_user is None:
		print("\n 1. SIGN UP")
		print("2. LOGIN")
		print("3. Exit")

		choice = input("Enter Choice:")
		if choice == "1":
			signup(users)
		elif choice == "2":
			current_user = login(users)
		elif choice == "3":
			break
		else:
			print("Invalid Choice")

	# USER LOGGED IN
	else:
		question = input(f"\n {current_user}:").strip()
		
		if question.lower() == "logout":
			print("Logged Out")
			current_user = None
		elif question.lower() == "exit":
			break
		elif question:
			name = users[current_user]["name"]
			
		try:
    			response = chatbot.invoke(
        		{
            			"name": name,
            			"question": question
        		},
        		config={
            			"configurable": {
                		"session_id": current_user
            			}
        		}
    			)

    			print("\nAI:", response.content)

		except Exception as error:
    			print("Error:", error)



