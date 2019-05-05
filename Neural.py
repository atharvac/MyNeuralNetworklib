import numpy as np
from math import e
import pickle

def sigmoid(x):
    return 1 / (1 + e**(-x))

class NeuralNetwork:
    """Number of input nodes, Number of hidden nodes, Number of output nodes.
    guess method returns only output while taking input(python list)"""
    def __init__(self,numI,numH,numO):
        self.input_nodes = numI
        self.hidden_nodes = numH
        self.output_nodes = numO

        self.weights_ih = np.random.rand(self.hidden_nodes,self.input_nodes)
        self.weights_ih = (self.weights_ih * 2) - 1
        self.weights_ho = np.random.rand(self.output_nodes,self.hidden_nodes)
        self.weights_ho = (self.weights_ho * 2) - 1

        #self.bias_h = 1
        #self.bias_o = 1
        self.bias_h = np.random.rand(self.hidden_nodes,1)
        self.bias_o = np.random.rand(self.output_nodes,1)
    
    #Input is python list
    def feedforward(self,input1):
        ############Generating hidden O/P##################
        input_mat = np.array(input1)
        input_mat = np.reshape(input_mat,(len(input1),1))
        hidden = self.weights_ih.dot(input_mat)
        hidden = hidden + self.bias_h
        ################Activation func####################
        hidden = sigmoid(hidden)
        ###################################################
        output = self.weights_ho.dot(hidden)
        output = output + self.bias_o
        ################Activation func####################
        output = sigmoid(output)
        #output = np.reshape(output,(1,self.output_nodes))
        #rr = output.tolist()
        #return rr[0]
        return input_mat,hidden,output

    #Inputs, answers is python list, LR (learning rate) is float
    def train(self,inputs,answers,LR):
        targets = np.array(answers)
        targets = np.reshape(targets,(len(answers),1))
        inp_mat,hiddens,outputs = self.feedforward(inputs)
        #Targets-Outputs
        output_errors = targets - outputs
        gradient_o = outputs * (1 - outputs)
        gradient_o = gradient_o * output_errors
        gradient_o = gradient_o * LR
        self.bias_o += gradient_o
        #Out deltas
        hidden_T = hiddens.transpose()
        weights_ho_delta = gradient_o.dot(hidden_T)
        self.weights_ho  += weights_ho_delta
        #Hidden errors
        weights_ho_inv = self.weights_ho.transpose()
        hidden_errors = weights_ho_inv.dot(output_errors)
        gradient_h = hiddens * (1 - hiddens)
        gradient_h = gradient_h * hidden_errors
        gradient_h = gradient_h * LR
        self.bias_h += gradient_h
        #Hidden deltas
        input_T = inp_mat.transpose()
        weights_ih_delta = gradient_h.dot(input_T)
        self.weights_ih += weights_ih_delta

    #inp is python list
    def guess(self,inp):
        i,h,o = self.feedforward(inp)
        o = np.reshape(o,(1,self.output_nodes))
        rr = o.tolist()
        return rr[0]

    @staticmethod
    def save(model,model_name):
        with open(model_name,'wb') as file:
            pickle.dump(model,file)

    @staticmethod
    def load(model_name):
        with open(model_name,'rb') as file:
            model = pickle.load(file)
        return model

#######################################################
def main():
    training_inputs = [[0,0],[0,1],[1,0],[1,1]]
    training_target = [[0,0],[1,0],[1,0],[0,1]]

    model = NeuralNetwork(2,4,2)

    for x in range(10000):
        for i in range(4):
            model.train(training_inputs[i],training_target[i],0.1)

    print(model.guess(training_inputs[0]))
    print(model.guess(training_inputs[1]))
    print(model.guess(training_inputs[2]))
    print(model.guess(training_inputs[3]))




if __name__ == '__main__':
    main()
