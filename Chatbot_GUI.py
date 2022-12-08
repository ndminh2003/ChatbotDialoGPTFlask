from flask import Flask, render_template, request
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# download and cache tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
# download and cache pre-trained model
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")


class ChatBot:
    # initialize
    def __init__(self):
        self.new_user_input_ids = None
        # once chat starts, the history will be stored for chat continuity
        self.chat_history_ids = None
        # make input ids global to use them anywhere within the object
        self.bot_input_ids = None
        # a flag to check whether to end the conversation
        self.end_chat = False

    def user_input(self):
        # receive input from user
        self.new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    def bot_response(self):
        # append the new user input tokens to the chat history
        # if chat has already begun
        if self.chat_history_ids is not None:
            self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1)
        else:
            # if first entry, initialize bot_input_ids
            self.bot_input_ids = self.new_user_input_ids
        # define the new chat_history_ids based on the preceding chats
        # generated a response while limiting the total chat history to 1000 tokens,
        self.chat_history_ids = model.generate(self.bot_input_ids, max_length=1000,
                                               pad_token_id=tokenizer.eos_token_id)
        # last output tokens from bot
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0],
                                    skip_special_tokens=True)
        # in case, bot fails to answer
        if response == "":
            response = self.random_response()
        # print bot response
        return '🤖 Bot: ' + response

    # in case there is no response from model
    def random_response(self):
        i = -1
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0],
                                    skip_special_tokens=True)
        # iterate over history backwards to find the last token
        while response == '':
            i = i - 1
            response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0],
                                        skip_special_tokens=True)
        # if it is a question, answer suitably
        if response.strip() == '?':
            reply = np.random.choice(["I don't know",
                                      "I am not sure",
                                      "No idea", "I will punch you if you say that again"
                                      ])
        # not a question? answer suitably
        else:
            reply = np.random.choice(["Great",
                                      "Fine. What's up?",
                                      "Okay", "The f**k", "Wanna die bruh ?"
                                      ])
        return reply


bot = ChatBot()

# GUI

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    global text
    text = request.args.get('msg')
    bot.user_input()
    response = bot.bot_response()
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=54321)