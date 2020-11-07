from keras.layers import Input, Activation, Dense,Flatten, BatchNormalization, Add, Conv2D,GlobalAveragePooling1D
from keras.layers import MaxPooling2D,AveragePooling2D,Permute,Reshape,LSTM,Lambda,GRU,Bidirectional,BatchNormalization,Concatenate
from keras import regularizers
from keras.optimizers import Adam
from models.attention_layer import *
from keras.models import Model
from utils import sharpe_ratio_loss,sharpe_ratio
from models.norm_layer import normalize_layer
###############################
# additive attention RNN models
###############################
def build_lstm_sa_aa_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']
    input_shape = params['input_shape']
    ts = input_shape[1]
    tickers = input_shape[0]
    type_norm = params['type_norm']

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    batch_norm = normalize_layer('layer')(reshape_inp)


    recurrent_layer,state_h,state_c = LSTM(units = units,
                    activation = activation,
                  kernel_regularizer=regularizers.l2(reg1),return_sequences=True,return_state=True) (batch_norm)


    batch_norm_2 = normalize_layer('layer')(recurrent_layer)


    ##ATTENTION LAYER
    # contxt_layer = AdditiveAttentionLayer(name='Att',latent_dim=32,kernel_regularizer=regularizers.l2(0.01))([batch_norm,batch_norm_2])

    contxt_layer = AdditiveAttentionLayer(name='Att',latent_dim=32,kernel_regularizer=regularizers.l2(0.01))([batch_norm_2,state_c])
    merge = Concatenate()([state_c,contxt_layer])


    out = Dense(units, kernel_regularizer =regularizers.l2(reg2),activation='tanh') (merge)
    batch_norm_3 = BatchNormalization()(out)


    out = Dense(tickers, kernel_regularizer =regularizers.l2(reg2)) (batch_norm_3)

    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])
    return model

def build_gru_sa_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']
    input_shape = params['input_shape']
    ts = input_shape[1]
    tickers = input_shape[0]
    type_norm = params['type_norm']

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    batch_norm = normalize_layer(type_norm)(reshape_inp)


    recurrent_layer,state_h = GRU(units = units,
                    activation = activation,
                  kernel_regularizer=regularizers.l2(reg1),return_sequences=True,return_state=True) (batch_norm)


    batch_norm_2 = normalize_layer(type_norm)(recurrent_layer)

    ##ATTENTION LAYER
    prob = SelfAttentionLayer(latent_dim=32,name='Att',kernel_regularizer=regularizers.l2(1e-4))(batch_norm_2)

    att = Lambda(lambda x: K.batch_dot(x[0],x[1]) ) ([prob,batch_norm_2])

    out = GlobalAveragePooling1D()(att)

    out = Dense(tickers, kernel_regularizer =regularizers.l2(reg2)) (out)

    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])
    return model

def build_lstm_sa_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']
    input_shape = params['input_shape']
    ts = input_shape[1]
    tickers = input_shape[0]
    type_norm = params['type_norm']

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    batch_norm = normalize_layer(type_norm)(reshape_inp)


    recurrent_layer,state_h,state_c = LSTM(units = units,
                    activation = activation,
                  kernel_regularizer=regularizers.l2(reg1),return_sequences=True,return_state=True) (batch_norm)


    batch_norm_2 = normalize_layer(type_norm)(recurrent_layer)

    ##ATTENTION LAYER
    prob = SelfAttentionLayer(latent_dim=32,name='Att',kernel_regularizer=regularizers.l2(1e-4))(batch_norm_2)

    att = Lambda(lambda x: K.batch_dot(x[0],x[1]) ) ([prob,batch_norm_2])

    out = GlobalAveragePooling1D()(att)

    out = Dense(tickers, kernel_regularizer =regularizers.l2(reg2)) (out)

    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])
    return model