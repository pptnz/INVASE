'''
Written by Jinsung Yoon
Date: Jan 1th 2019
INVASE: Instance-wise Variable Selection using Neural Networks Implementation on Synthetic Datasets
Reference: J. Yoon, J. Jordon, M. van der Schaar, "IINVASE: Instance-wise Variable Selection using Neural Networks," International Conference on Learning Representations (ICLR), 2019.
Paper Link: https://openreview.net/forum?id=BJg_roAcK7
Contact: jsyoon0823@g.ucla.edu

---------------------------------------------------

Instance-wise Variable Selection (INVASE) - with baseline networks
'''

#%% Necessary packages
# 1. Keras
from keras.layers import Input, Dense, Multiply
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

# 2. Others
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

#%% Define PVS class
class PVS():
    
    # 1. Initialization
    '''
    x_train: training samples
    data_type: Syn1 to Syn 6
    '''
    def __init__(self, x_train):
        self.latent_dim1 = 100      # Dimension of actor (generator) network
        self.latent_dim2 = 200      # Dimension of critic (discriminator) network
    def __init__(self, x_train, load_model=False):
        
        self.batch_size = 100       # Batch size
        self.epochs = 20000          # Epoch size (large epoch is needed due to the policy gradient framework)
        self.lamda = 0.1            # Hyper-parameter for the number of selected features

        self.input_shape = x_train.shape[1]     # Input dimension
        self.output_size = 4

        # Actionvation.
        self.activation = 'selu'

        # Use Adam optimizer with learning rate = 0.0001
        optimizer = Adam(0.00001)
        
        if load_model:
            self.load_models()
        else:
            # Build and compile the discriminator (critic)
            self.discriminator = self.build_discriminator()
            # Use categorical cross entropy as the loss
            self.discriminator.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

            # Build the generator (actor)
            self.generator = self.build_generator()
            # Use custom loss (my loss)
            self.generator.compile(loss=self.my_loss, optimizer=optimizer)

            # Build and compile the value function
            self.valfunction = self.build_valfunction()
            # Use categorical cross entropy as the loss
            self.valfunction.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    #%% Custom loss definition
    def my_loss(self, y_true, y_pred):
        
        # dimension of the features
        d = y_pred.shape[1]        

        # Put all three in y_true 
        # 1. selected probability
        sel_prob = y_true[:,:d]
        # 2. discriminator output
        dis_prob = y_true[:,d:(d + self.output_size)]
        # 3. valfunction output
        val_prob = y_true[:, (d + self.output_size):(d + 2 * self.output_size)]
        # 4. ground truth
        y_final = y_true[:, (d + 2 * self.output_size):]

        # A1. Compute the rewards of the actor network
        Reward1 = tf.reduce_sum(y_final * tf.log(dis_prob + 1e-8), axis = 1)  
        
        # A2. Compute the rewards of the actor network
        Reward2 = tf.reduce_sum(y_final * tf.log(val_prob + 1e-8), axis = 1)  

        # Difference is the rewards
        Reward = Reward1 - Reward2

        # B. Policy gradient loss computation. 
        loss1 = Reward * tf.reduce_sum( sel_prob * K.log(y_pred + 1e-8) + (1-sel_prob) * K.log(1-y_pred + 1e-8), axis = 1) - self.lamda * tf.reduce_mean(y_pred, axis = 1)
        
        # C. Maximize the loss1
        loss = tf.reduce_mean(-loss1)

        return loss

    #%% Generator (Actor)
    def build_generator(self):

        model = Sequential()
        
        model.add(Dense(100, activation=self.activation, name = 's/dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape))
        model.add(Dense(self.input_shape, activation = 'sigmoid', name = 's/dense2', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()

        feature = Input(shape=(self.input_shape,), dtype='float32')
        select_prob = model(feature)

        return Model(feature, select_prob)

    #%% Discriminator (Critic)
    def build_discriminator(self):

        model = Sequential()
                
        model.add(Dense(200, activation=self.activation, name = 'dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape)) 
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(self.output_size, activation ='softmax', name ='dense2', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()
        
        # There are two inputs to be used in the discriminator
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')
        # 2. Selected Features
        select = Input(shape=(self.input_shape,), dtype='float32')         
        
        # Element-wise multiplication
        model_input = Multiply()([feature, select])
        prob = model(model_input)

        return Model([feature, select], prob)
        
    #%% Value Function
    def build_valfunction(self):

        model = Sequential()
                
        model.add(Dense(200, activation=self.activation, name = 'v/dense1', kernel_regularizer=regularizers.l2(1e-3), input_dim = self.input_shape)) 
        model.add(BatchNormalization())     # Use Batch norm for preventing overfitting
        model.add(Dense(self.output_size, activation ='softmax', name = 'v/dense2', kernel_regularizer=regularizers.l2(1e-3)))
        
        model.summary()
        
        # There are one inputs to be used in the value function
        # 1. Features
        feature = Input(shape=(self.input_shape,), dtype='float32')       
        
        # Element-wise multiplication
        prob = model(feature)

        return Model(feature, prob)

    #%% Sampling the features based on the output of the generator
    def Sample_M(self, gen_prob):
        
        # Shape of the selection probability
        n = gen_prob.shape[0]
        d = gen_prob.shape[1]
                
        # Sampling
        samples = np.random.binomial(1, gen_prob, (n,d))
        
        return samples

    #%% Training procedure
    def train(self, x_train, y_train):

        # For each epoch (actually iterations)
        for epoch in range(self.epochs):

            #%% Train Discriminator
            # Select a random batch of samples
            idx = np.random.randint(0, x_train.shape[0], self.batch_size)
            x_batch = x_train[idx,:]
            y_batch = y_train[idx,:]

            # Generate a batch of probabilities of feature selection
            gen_prob = self.generator.predict(x_batch)
            
            # Sampling the features based on the generated probability
            sel_prob = self.Sample_M(gen_prob)     
            
            # Compute the prediction of the critic based on the sampled features (used for generator training)
            dis_prob = self.discriminator.predict([x_batch, sel_prob])

            # Train the discriminator
            d_loss = self.discriminator.train_on_batch([x_batch, sel_prob], y_batch)

            #%% Train Valud function

            # Compute the prediction of the critic based on the sampled features (used for generator training)
            val_prob = self.valfunction.predict(x_batch)

            # Train the discriminator
            v_loss = self.valfunction.train_on_batch(x_batch, y_batch)
            
            #%% Train Generator
            # Use three things as the y_true: sel_prob, dis_prob, and ground truth (y_batch)
            y_batch_final = np.concatenate( (sel_prob, np.asarray(dis_prob), np.asarray(val_prob), y_batch), axis = 1 )

            # Train the generator
            g_loss = self.generator.train_on_batch(x_batch, y_batch_final)

            #%% Plot the progress
            dialog = 'Epoch: ' + str(epoch) + ', d_loss (Acc): ' + str(d_loss[1]) + ', v_loss (Acc): ' + str(v_loss[1]) + ', g_loss: ' + str(np.round(g_loss,4))

            if epoch % 100 == 0:
                print(dialog)
    
    #%% Selected Features        
    def output(self, x_train):
        
        gen_prob = self.generator.predict(x_train)
        
        return np.asarray(gen_prob)
     
    #%% Prediction Results 
    def get_prediction(self, x_train, m_train):
        
        val_prediction = self.valfunction.predict(x_train)
        
        dis_prediction = self.discriminator.predict([x_train, m_train])
        
        return np.asarray(val_prediction), np.asarray(dis_prediction)

    def save_models(self):
        self.generator.save('./model/generator.h5')
        self.valfunction.save('./model/valfunction.h5')
        self.discriminator.save('./model/discriminator.h5')

    def load_models(self):
        self.generator = load_model('./model/generator.h5', custom_objects={'my_loss': self.my_loss})
        self.valfunction = load_model('./model/valfunction.h5')
        self.discriminator = load_model('./model/discriminator.h5')


#%% Main Function
if __name__ == '__main__':
        
    # Data generation function import
    from Data_Reader import read_data

    x_train, y_train, x_test, y_test = read_data(source="./data/pathway_activity.csv")

    #%% 
    # 1. PVS Class call
    PVS_Alg = PVS(x_train, load_model=True)
    
    # 2. Algorithm training
    PVS_Alg.train(x_train, y_train)
    
    # 3. Get the selection probability on the testing set
    Sel_Prob_Test = PVS_Alg.output(x_test)
    
    # 4. Selected features
    score = 1.*(Sel_Prob_Test > 0.5)
    num_sel_features = score.sum() / len(score)
    print("Number of Selected Features:", num_sel_features)

    # np.savetxt("testset_selected_feature.csv", score, delimiter=',')
    PVS_Alg.save_models()

    # 5. Prediction
    val_predict, dis_predict = PVS_Alg.get_prediction(x_test, score)

    val_accuracy_table = [[0 for _ in range(PVS_Alg.output_size)] for _ in range(PVS_Alg.output_size)]
    dis_accuracy_table = [[0 for _ in range(PVS_Alg.output_size)] for _ in range(PVS_Alg.output_size)]

    val_label = np.argmax(val_predict, axis=1)
    dis_label = np.argmax(val_predict, axis=1)
    true_label = np.argmax(y_test, axis=1)

    for i in range(len(true_label)):
        val_accuracy_table[val_label[i]][true_label[i]] += 1
        dis_accuracy_table[dis_label[i]][true_label[i]] += 1

    # Compute weighted F1 score
    from sklearn.metrics import f1_score
    val_f1_score = f1_score(true_label, val_label, average='weighted')
    dis_f1_score = f1_score(true_label, dis_label, average='weighted')

    # Compute Accuracy
    val_correct = 0
    dis_correct = 0

    for i in range(PVS_Alg.output_size):
        val_correct += val_accuracy_table[i][i]
        dis_correct += dis_accuracy_table[i][i]

    val_accuracy = val_correct / len(val_label)
    dis_accuracy = dis_correct / len(dis_label)

    # Print Accuracy Table
    print("\nBaseline Prediction")
    for i in range(len(val_accuracy_table)):
        for j in range(len(val_accuracy_table[i])):
            print(val_accuracy_table[i][j], end=' ')
        print()
    print("Weighted F1 Score: {:.4f}".format(val_f1_score))
    print("Accuracy: {:.4f}".format(val_accuracy))

    print("\nPredictor Prediction")
    for i in range(len(dis_accuracy_table)):
        for j in range(len(dis_accuracy_table[i])):
            print(dis_accuracy_table[i][j], end=' ')
        print()
    print("Weighted F1 Score: {:.4f}".format(dis_f1_score))
    print("Accuracy: {:.4f}".format(dis_accuracy))
