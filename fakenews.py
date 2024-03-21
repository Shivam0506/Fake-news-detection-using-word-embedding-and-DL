import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Use 'tensorflow.keras' here
from tensorflow.keras.preprocessing.text import Tokenizer  # Use 'tensorflow.keras' here
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
nltk.download('punkt')  # This is needed for NLTK to work
from tensorflow.keras.models import load_model
import string
# # Load the pickled model
# with open('cnnfasttext_model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)
model = load_model("best_model.h5")
# Assuming you already have 'new_input_data' and 'sequence_length'
# Tokenize and preprocess the new input data as you did before
new_input_data = [ "BREAKING : Trump Expressed Concern Over Anthony Weinerâ€™s â€œIllegal Accessâ€ to Classified Info 2 Months ago BREAKING : Trump Expressed Concern Over Anthony Weinerâ€™s â€œIllegal Accessâ€ to Classified Info 2 Months ago Breaking News By Amy Moreno October 28, 2016. Once again, Trump was right. Back in August, in a statement regarding Hillaryâ€™s carelessness handling classified documents, Trump stated that he was concerned that Weiner had â€œaccessâ€ to information he shouldnâ€™t. Now that weâ€™re learning that the FBI discovered â€œnew emailsâ€ on a â€œdeviceâ€ associated to Weiner, it looks as if Trump was right AGAIN. â€” Deplorable AJ (@asamjulian) October 28, 2016 This is a movement â€“ we are the political OUTSIDERS fighting against the FAILED GLOBAL ESTABLISHMENT! Join the resistance and help us fight to put America First! Amy Moreno is a Published Author , Pug Lover & Game of Thrones Nerd. You can follow her on Twitter here and Facebook here . Support the Trump Movement and help us fight Liberal Media Bias. Please LIKE and SHARE this story on Facebook or Twitter.  "]

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','', text) #remove text enclosed in square brackets, including the brackets themselves
    text = re.sub("\\W", " ", text) #replaces all non-word characters (e.g., punctuation, special characters, symbols) with spaces
    text = re.sub('https?://\S+|www\.\S+', '', text) # removes URLs from the text by matching and removing both HTTP/HTTPS URLs and "www" URLs
    text = re.sub('<.*?>+', '', text) #remove HTML tags and their contents
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) #remove all punctuation characters
    text = re.sub('\n', '', text) #removes newline characters, which are typically used to represent line breaks or paragraphs in text
    text = re.sub('\w*\d\w', '', text) #removes words containing numbers or alphanumeric patterns
    return text

new_sequences = [wordopt(sentence) for sentence in new_input_data]

from keras.preprocessing.sequence import pad_sequences
tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(new_sequences)  # Assuming x_train is a list of text strings
new_sequences = tokenizer1.texts_to_sequences(new_sequences)

# Pad sequences
new_sequences = pad_sequences(new_sequences, maxlen=8280)

# Convert 'new_sequences' to a NumPy array
new_data = np.array(new_sequences)

# Pad or truncate the sequences to match the sequence length
new_data = pad_sequences(new_data, maxlen=8280, padding='post', truncating='post')

# Make predictions on the new data
predictions = model.predict(new_data)

# The 'predictions' array will contain probability scores for each class (0 and 1)
# You can convert these scores to class labels based on a threshold (e.g., 0.5)
predicted_labels = [1 if score >= 0.5 else 0 for score in predictions]

class_mapping = {0: "fake", 1: "true"}

# Use the mapping to transform the predicted labels
predicted_class_names = [class_mapping[label] for label in predicted_labels]

print(predicted_class_names)
