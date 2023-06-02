from transformers import TFBertModel,BertTokenizer
import tensorflow as tf

def load_models():
    tokenizerbrt = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    models = {}
    # Define the custom object scope
    custom_objects = {'TFBertModel': TFBertModel}  # Replace TFBertModel with the actual custom layer class
    # Load the model architecture from the saved file
    hate_classification_model = tf.keras.models.load_model('hate_classification_model/model.h5', custom_objects=custom_objects)
    # Load the model weights into the architecture
    hate_classification_model.load_weights('hate_classification_model/model_weights.h5')
    models['dz'] = hate_classification_model
    # Load the model architecture from the saved file
    hate_classification_model_en = tf.keras.models.load_model('hate_classification_model_eng/model_en.h5', custom_objects=custom_objects) # english model
    # Load the model weights into the architecture
    hate_classification_model_en.load_weights('hate_classification_model_eng/model_weights_en.h5')
    models['en'] = hate_classification_model_en
    return models, tokenizerbrt

models ,tokenizerbrt = load_models()