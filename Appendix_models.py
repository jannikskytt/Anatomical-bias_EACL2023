import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import TFElectraModel

#RNN model, ~4M parameters
def build_rnn(n_class=2):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, 
                        input_length=None, 
                        weights=[embedding_matrix], #Embedding matrix is 300-dimensional FastText embeddings
                        trainable=False))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(n_class-1, activation='sigmoid'))
    return model



# ELECTRA model, ~13M parameters
class HF_ClassificationModel(tf.keras.Model):
   def __init__(self, num_labels=None, model_name=da_electra):
       '''
       Hyperparametre modeleret efter HF
       '''
       super().__init__(name="EHR_classification")
       self.electra = TFElectraModel.from_pretrained(model_name, from_pt=False)
       self.dense = tf.keras.layers.Dense(units=256, kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02))
       self.dropout = tf.keras.layers.Dropout(rate=0.1)
       self.out_proj = tf.keras.layers.Dense(num_labels-1, kernel_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02))

   def call(self, inputs, **kwargs):
       x = self.electra(inputs, **kwargs)
       x = x[0][:,0,:] 
       x = self.dropout(x)
       x = self.dense(x)
       x = tf.keras.activations.gelu(x)
       x = self.dropout(x)
       return tf.nn.sigmoid(self.out_proj(x))