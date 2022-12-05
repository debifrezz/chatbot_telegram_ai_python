import numpy as np
import json
import os

from telegram.ext import *
from util import JSONParser
from util import Preprocess
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


# Load Token
with open('token.txt', 'r') as r:
    TOKEN = r.read()

# Text input user
text = {}

# load data
path = 'data/data.json'
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()

# Preprocessing
preproces = Preprocess()

df['text_input_prep'] = df.text_input.apply(preproces.preprocess)
# Modeling
pipeline = make_pipeline(CountVectorizer(),MultinomialNB())
# Training
print("[INFO!!!] Training Data..")
pipeline.fit(df.text_input_prep, df.intents)


def bot_response(text, pipeline, jp):
    text = preproces.preprocess(text)
    res = pipeline.predict_proba([text])
    max_prob = max(res[0])
    if max_prob < 0.4:
        return "Maaf saya ga ngerti", None
    else:
        max_id = np.argmax(res[0])
        pred_tag = pipeline.classes_[max_id]
        return jp.get_response(pred_tag), pred_tag


def start(update, context):
    update.message.reply_text("Welcome!")

def help(update, context):
    update.message.reply_text("""
    /start - Start Converstation
    /help - Show this message
    """)

def handle_message(update, context):
    text = update.message.text
    res, tag = bot_response(text, pipeline, jp)
    update.message.reply_text(f"{res}")
    with open("data/input.json") as d:
        data = json.load(d)
        temp = data["intents"]
        saved = {"tag":f"{tag}","patterns":[f"{text}"],"responses":[f"{res}"]}
        temp.append(saved)
    write_json(data)

def write_json(data, filename="data/input.json"):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def shutdown(update, context):
    update.message.reply_text("server shutdown")
    os.kill(os.getpid(), 9)



updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("shutdown", shutdown))
dp.add_handler(MessageHandler(Filters.all, callback = handle_message))
updater.start_polling()
updater.idle()