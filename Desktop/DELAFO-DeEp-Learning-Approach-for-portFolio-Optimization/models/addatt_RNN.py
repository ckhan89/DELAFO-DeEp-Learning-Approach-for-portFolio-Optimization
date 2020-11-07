from keras.layers import Input, Activation, Dense,Flatten, BatchNormalization, Add, Conv2D
from keras.layers import MaxPooling2D,AveragePooling2D,Permute,Reshape,LSTM,Lambda,GRU,Bidirectional,BatchNormalization,Concatenate
from keras import regularizers
from keras.optimizers import Adam,SGD
from models.attention_layer import *
from keras.models import Model
from utils import sharpe_ratio_loss,sharpe_ratio
from models.norm_layer import normalize_layer
import pdb
###############################
# additive attention RNN models
###############################
def build_add_att_lstm_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']
    input_shape = params['input_shape']
    type_norm = params['type_norm']

    ts = input_shape[1]
    tickers = input_shape[0]

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    batch_norm = normalize_layer(type_norm)(reshape_inp)


    recurrent_layer = LSTM(units = units,
                    activation = activation,
                  kernel_regularizer=regularizers.l2(reg1)) (batch_norm)

    batch_norm_2 = normalize_layer(type_norm)(recurrent_layer)



    ##ATTENTION LAYER
    contxt_layer = AdditiveAttentionLayer(name='Att',latent_dim=32,kernel_regularizer=regularizers.l2(0.01))([batch_norm,batch_norm_2])

    merge = Concatenate()([batch_norm_2,contxt_layer])

    out = Dense(units, kernel_regularizer =regularizers.l2(reg2),activation='tanh') (merge)
    batch_norm_3 = normalize_layer(type_norm)(out)


    out = Dense(tickers, kernel_regularizer =regularizers.l2(reg2)) (batch_norm_3)

    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])
    return model

def build_add_att_gru_model(params):
    units = params['units']
    activation = params['activation']
    reg1 = params['l2']
    reg2 = params['l2_1']
    lr = params['l2_2']
    input_shape = params['input_shape']
    type_norm = params['type_norm']

    ts = input_shape[1]
    tickers = input_shape[0]

    input = Input(shape=input_shape)
    reshape_inp = Lambda(lambda x: K.permute_dimensions(x,pattern=(0,2,1,3))) (input)
    reshape_inp = Reshape((ts,-1)) (reshape_inp)

    # batch_norm = BatchNormalization()(reshape_inp)
    batch_norm = normalize_layer(type_norm)(reshape_inp)

    recurrent_layer = GRU(units = units,
                    activation = activation,
                  kernel_regularizer=regularizers.l2(reg1)) (batch_norm)

    # batch_norm_2 = BatchNormalization()(recurrent_layer)
    batch_norm_2 = normalize_layer(type_norm)(recurrent_layer)
    ##ATTENTION LAYER
    contxt_layer = AdditiveAttentionLayer(name='Att',latent_dim=32,kernel_regularizer=regularizers.l2(0.01))([batch_norm,batch_norm_2])
    # pdb.set_trace()
    ### old model#############
    merge = Concatenate()([batch_norm_2,contxt_layer])

    out = Dense(units, kernel_regularizer =regularizers.l2(reg2),activation='tanh') (merge)
    batch_norm_3 = normalize_layer(type_norm)(out)
    out = Dense(tickers, kernel_regularizer =regularizers.l2(reg2)) (batch_norm_3)
    ######################
    ## change a bit
    # out = Dense(units, kernel_regularizer =regularizers.l2(reg2),activation='tanh') (contxt_layer)
    # merge = Concatenate()([batch_norm_2,out])
    # batch_norm_3 = normalize_layer(type_norm)(merge)
    # out = Dense(tickers, kernel_regularizer =regularizers.l2(reg2)) (batch_norm_3)
    # #### finish change
    out = Activation('sigmoid')(out)

    model = Model([input], [out])
    optimizer = Adam(lr = lr)
    # optimizer = Adam(0.1)
    model.compile(loss=sharpe_ratio_loss, optimizer=optimizer, metrics = [sharpe_ratio])
    return model

