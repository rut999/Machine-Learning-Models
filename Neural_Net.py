#!/usr/local/bin/python3
"""
Code by: nakopa,pvajja,rparvat
Authors: Naga Anjaneyulu , Prudhvi Vajja , Rutvik Parvataneni
"""
import numpy as np
import copy
# import time as t
import pickle

#Referred to this :: https://zhenye-na.github.io/2018/09/09/build-neural-network-with-mnist-from-scratch.html
class NeuralNet(object):
    def __init__(self,input_nodes, output_nodes, learning_rate, epochs):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_1 = 0
        self.weight_2 = 0
        self.weight_3 = 0
        self.weight_4 = 0
        self.bias_1 = 0
        self.bias_2 = 0
        self.bias_3 = 0
        self.bias_4 = 0

    def Normalize(self, x):
        y_mean = np.mean(x)
        y_std = np.std(x)
        x = (x - y_mean) / y_std
        return x

    def reLU_derivative(self, x):
        # x = self.Normalize(x)
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    def Cost_Loss(self, y, y_pred):
        n = y_pred.shape[1]
        cost = (1. / (2 * n)) * np.sum((y - y_pred) ** 2)
        return cost

    def c_Loss(self,Y, Y_hat):
        L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
        m = Y.shape[1]
        L = -(1. / m) * L_sum
        return L

    def Dropout(self, x, drop_prob):
        keep_prob = 1 - drop_prob
        mask = np.random.binomial(1, drop_prob, size=x.shape)
        if keep_prob > 0.0:
            scaling = (1/keep_prob)
        else:
            scaling = 0.0
        return mask*x*scaling

    def forward_prop(self,x_train):
        self.layer1 = reLU(np.dot(x_train, self.weight_1) + self.bias_1)
        # self.layer1 = Dropout(self.layer1, self.drop_prob)
        self.layer2 = reLU(np.dot(self.layer1, self.weight_2) + self.bias_2)
        self.layer3 = reLU(np.dot(self.layer2, self.weight_3) + self.bias_3)
        # self.layer2 = Dropout(self.layer2, self.drop_prob)
        self.out = np.dot(self.layer3, self.weight_4) + self.bias_4
        self.out_final = soft_max(self.out)
        return self.out_final

    def fit(self):
        # np.random.seed(1)
        self.input_nodes = np.array(self.input_nodes, dtype=np.float)
        self.input_nodes/=255
        self.input_nodes = self.Normalize(self.input_nodes)
        #Initialize bias
        self.bias_1 = np.zeros((1,50))
        self.bias_2 = np.zeros((1,50))
        self.bias_3 = np.zeros((1, 50))
        self.bias_4 = np.zeros((1, self.output_nodes.shape[1]))

        #Weight Initialization ::
        # self.weight_1 = np.random.uniform(-1, 1, size=(self.input_nodes.shape[1], 50)) /np.sqrt(self.input_nodes.shape[1])
        # self.weight_2 = np.random.uniform(-1, 1, size=(50,50))/np.sqrt(50)
        # self.weight_3 = np.random.uniform(-1, 1, size=(50, 50)) / np.sqrt(50)
        # self.weight_4 = np.random.uniform(-1, 1, size=(50, self.output_nodes.shape[1]))/np.sqrt(50)
        # self.weight_1 = np.random.rand(self.input_nodes.shape[1], 50) /np.sqrt(self.input_nodes.shape[1])
        # self.weight_2 = np.random.rand(50, 50)/np.sqrt(50)
        # self.weight_3 = np.random.rand(50, 50) / np.sqrt(50)
        # self.weight_4 = np.random.rand(50, self.output_nodes.shape[1]) /np.sqrt(50)
        self.weight_1 = np.random.randn(self.input_nodes.shape[1], 50) /np.sqrt(self.input_nodes.shape[1])
        self.weight_2 = np.random.randn(50, 50)/np.sqrt(50)
        self.weight_3 = np.random.randn(50, 50) / np.sqrt(50)
        self.weight_4 = np.random.randn(50, self.output_nodes.shape[1])/np.sqrt(50)


        #epoch values
        best_accuracy = 0
        w1, w2, w3, w4, b1, b2, b3, b4 = 0, 0, 0, 0, 0, 0, 0, 0
        #Splitting the data ::
        for epoch in range(self.epochs):
            for iteration in range(self.input_nodes.shape[0]):
                rand_ind = np.random.randint(0, self.input_nodes.shape[0])
                batch_x = self.input_nodes[rand_ind,: ].reshape(1, self.input_nodes.shape[1])
                batch_y = self.output_nodes[rand_ind, :].reshape(1, self.output_nodes.shape[1])

                #Forward Propagation ::
                self.layer1 = reLU(np.dot(batch_x, self.weight_1) + self.bias_1)
                self.layer2 = reLU(np.dot(self.layer1, self.weight_2) + self.bias_2)
                self.layer3 = reLU(np.dot(self.layer2, self.weight_3) + self.bias_3)
                self.out = np.dot(self.layer3, self.weight_4) + self.bias_4
                self.out_final = soft_max(self.out)
                #Back Propagation ::
                #del 4
                self.error_4 = self.out_final - batch_y
                self.w4_derv = (self.layer3.T).dot(self.error_4)
                # self.weight_3 -= self.learning_rate * (self.w3_derv.clip(min=0.000001))
                self.weight_4 -= self.learning_rate * (self.w4_derv)

                #del3
                self.error_3 = self.error_4.dot(self.weight_4.T) * self.reLU_derivative(self.layer3)
                self.w3_derv = self.layer2.T.dot(self.error_3)
                # self.weight_2 -= self.learning_rate *(self.w2_derv.clip(min=0.000001))
                self.weight_3 -= self.learning_rate * (self.w3_derv)

                #del 2
                self.error_2 = self.error_3.dot(self.weight_3.T) * self.reLU_derivative(self.layer2)
                self.w2_derv = self.layer1.T.dot(self.error_2)
                # self.weight_2 -= self.learning_rate *(self.w2_derv.clip(min=0.000001))
                self.weight_2 -= self.learning_rate * (self.w2_derv)

                #del 1
                self.error_1 = self.error_2.dot(self.weight_2.T) * self.reLU_derivative(self.layer1)
                self.w1_derv = batch_x.T.dot(self.error_1)
                # self.weight_1 -= self.learning_rate * (self.w1_derv.clip(min=0.000001))
                self.weight_1 -= self.learning_rate * (self.w1_derv)

                #Bias 1
                self.db1 = np.sum(self.error_1, axis = 0, keepdims=True)
                # self.bias_1 -= self.learning_rate * (self.db1.clip(min=0.000001))
                self.bias_1 -= self.learning_rate * (self.db1)

                #Bias2
                self.db2 = np.sum(self.error_2, axis=0,  keepdims=True)
                # self.bias_2 -= self.learning_rate * (self.db2.clip(min=0.000001))
                self.bias_2 -= self.learning_rate * (self.db2)

                #Bias 3
                self.b3_derv = np.sum(self.error_3, axis =0, keepdims=True)
                # self.bias_3 -= self.learning_rate * (self.b3_derv.clip(min=0.000001))
                self.bias_3 -= self.learning_rate * (self.b3_derv)

                #bias 4
                self.b4_derv = np.sum(self.error_4, axis = 0, keepdims=True)
                self.bias_4 -= self.learning_rate * self.b4_derv


            w = self.forward_prop(self.input_nodes)
            y_pred1 = np.argmax(w, axis=1) * 90
            class_labels = np.argmax(self.output_nodes, axis=1)*90
            accuracy = score_acc(class_labels,y_pred1)*100
            loss = self.c_Loss(self.output_nodes,self.out_final)
            print('Results ::',(epoch,self.learning_rate, loss, accuracy))

            #Storing the Weights
            if accuracy>best_accuracy:
                w1,w2,w3,w4 = copy.deepcopy(self.weight_1), copy.deepcopy(self.weight_2), copy.deepcopy(self.weight_3), copy.deepcopy(self.weight_4)
                b1, b2, b3, b4 = copy.deepcopy(self.bias_1), copy.deepcopy(self.bias_2), copy.deepcopy(self.bias_3), copy.deepcopy(self.bias_4)
                best_accuracy = copy.deepcopy(accuracy)
        #Forward prop ::
        self.out_final = self.forward_prop(self.input_nodes)
        y_pred = (self.out_final == self.out_final.max(axis=1)[:, None]).astype(int)
        y_pr = np.argmax(y_pred, axis=1) * 90
        output_n = np.argmax(self.output_nodes, axis=1) * 90
        accuracy = score_acc(output_n, y_pr) * 100
        if(accuracy>best_accuracy):
            best_accuracy= copy.deepcopy(accuracy)
        print("The Training  accuracy is  :: ", best_accuracy)
        return w1, w2, w3, w4, b1, b2, b3, b4

def score_acc(y, ypred):
    return ((ypred == y).astype(int).mean())

def reLU(y):
    return np.maximum(0, y)

def soft_max(x):
    # x = self.Normalize(x)
    expA = np.exp(x)
    return expA / expA.sum()

def Predict( Z, w1, w2, w3, w4, b1, b2, b3, b4):
    opt_1 = Z@w1+b1
    act_1 = reLU(opt_1)

    opt_2 = act_1@w2+b2
    act_2 = reLU(opt_2)

    opt_3 = act_2@w3+b3
    act_3 = reLU(opt_3)

    opt_4 = act_3@w4+b4
    soft = soft_max(opt_4)
    ypred = np.argmax(soft, axis=1) * 90
    return(ypred)

def read_file(filename):
    f = open(filename, "r")
    train_data = []
    lines = f.readlines()
    for x in lines:
        l = x.split()
        train_data.append(l[1:])
    f.close()
    return np.array(train_data)
#https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
def one_hot(A):
    a = np.array(A)
    a = a.astype(int)
    z = a.size
    y = int(a.max()+1)
    b = np.zeros((z, y))
    w = np.arange(a.size)
    b[w, a] = 1
    return b

def train(train_data,file_name):
    y_train, x_train = train_data[:, 0], train_data[:, 1:]
    x_train, y_train = x_train.astype(int), y_train.astype(int)
    y_train = y_train / 90
    y_train = one_hot(y_train)

    N_n = NeuralNet(x_train, y_train, learning_rate=0.0001, epochs=5)
    w1, w2, w3, w4, b1, b2, b3, b4 = N_n.fit()

    # saving weights & bias into a dictionary
    data_dict = {}
    data_dict['w1'], data_dict['w2'], data_dict['w3'], data_dict['w4'] = w1, w2, w3, w4
    data_dict['b1'], data_dict['b2'], data_dict['b3'], data_dict['b4'] = b1, b2, b3, b4

    # Referred to this :: https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
    with open(file_name, 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test(test_data,file_name):
    y_test, x_test = test_data[:, 0], test_data[:, 1:]
    # \y_test, x_test = train_data[:, 0], test_data[:, 1:]
    y_test, x_test = y_test.astype(int), x_test.astype(int)
    with open(file_name, 'rb') as handle:
        z = pickle.load(handle)

    W1, W2, W3, W4 = z['w1'], z['w2'], z['w3'], z['w4']
    B1, B2, B3, B4 = z['b1'], z['b2'], z['b3'], z['b4']

    y_test_p = Predict(x_test, W1, W2, W3, W4, B1, B2, B3, B4)

    # accuracy_test = accuracy_score(y_test, y_test_p)
    accuracy_test = score_acc(y_test, y_test_p)
    print("Test Accuracy ", accuracy_test)
    # print("Overall Time taken is ::", t.time() - start_time)
    return y_test_p

# if __name__ == "__main__":
#     start_time = t.time()
#     train_data = read_file('train-data.txt')
#     y_train, x_train = train_data[:, 0], train_data[:, 1:]
#     test_data = read_file('./test-data.txt')
#     y_test, x_test = test_data[:, 0], test_data[:, 1:]
#
#     x_train,  y_train  = x_train.astype(int), y_train.astype(int)
#     y_train = y_train/90
#     y_train = one_hot(y_train)
#     y_test, x_test = y_test.astype(int), x_test.astype(int)
#
#
#     N_n = NeuralNet(x_train, y_train, learning_rate= 0.0001,epochs= 5)
#     w1, w2, w3, w4, b1, b2, b3, b4 = N_n.fit()
#     #saving weights & bias into a dictionary
#     data_dict = {}
#     data_dict['w1'], data_dict['w2'], data_dict['w3'], data_dict['w4'] = w1, w2, w3, w4
#     data_dict['b1'], data_dict['b2'], data_dict['b3'], data_dict['b4'] = b1, b2, b3, b4
#
#     #Referred to this :: https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
#     with open('model_file.txt','wb') as handle:
#         pickle.dump(data_dict,handle,protocol=pickle.HIGHEST_PROTOCOL)
#
#     with open('model_file.pickle','rb') as handle:
#         z = pickle.load(handle)
#
#     W1,W2,W3,W4 = z['w1'], z['w2'], z['w3'], z['w4']
#     B1,B2,B3,B4 = z['b1'], z['b2'], z['b3'], z['b4']
#
#     y_test_p = N_n.Predict(x_test,W1, W2, W3, W4, B1, B2, B3, B4)
#
#     # accuracy_test = accuracy_score(y_test, y_test_p)
#     accuracy_test = N_n.score_acc(y_test, y_test_p)
#     print("Test Accuracy ",accuracy_test)
#     print("Overall Time taken is ::",t.time() - start_time)

















