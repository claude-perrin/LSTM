import pandas as pd              
import matplotlib.pyplot as plt  
from activation_functions import *



class LSTM:
    def __init__(self, ltm, stm, input):
        self.input = input
        self.ltm = ltm
        self.stm = stm

#        #Long-term to remember Forget Gate
        self.w_forget_gate = { "w_stm": 2.7, "w_in": 1.63, "bias": 1.62 }
#       

        #Poterntial Memory to remember ltm InputGate
        self.w_input_gate_sigmoid = { "w_stm": 2, "w_in": 1.65, "bias": 0.62 } 
#        #Potential Long-Term memory InputGate
        self.w_input_gate_tan = { "w_stm": 1.41, "w_in": 0.94, "bias": -0.32 }

#        #Potential memory to remember stm OutputGate
        self.w_output_gate = { "w_stm": 4.38, "w_in": -0.19, "bias": 0.59 }

    def forget_gate(self):
       w_stm, w_in, bias = self.w_forget_gate["w_stm"], self.w_forget_gate["w_in"], self.w_forget_gate["bias"]
       forget_percentage = sigmoid(self.stm * w_stm + self.input * w_in + bias)
       self.ltm = self.ltm * forget_percentage


    def input_gate(self): 
       w_stm_sig, w_in_sig, bias_sig = self.w_input_gate_sigmoid["w_stm"], self.w_input_gate_sigmoid["w_in"], self.w_input_gate_sigmoid["bias"]
       w_stm_tan, w_in_tan, bias_tan = self.w_input_gate_tan["w_stm"], self.w_input_gate_tan["w_in"], self.w_input_gate_tan["bias"]
       
       percentage_ltm = sigmoid(self.stm * w_stm_sig + self.input * w_in_sig + bias_sig)
       potential_ltm = tanh_activation(self.stm * w_stm_tan + self.input * w_in_tan + bias_tan)
       
       self.ltm = percentage_ltm * potential_ltm + self.ltm

    
    def output_gate(self): 
       w_stm, w_in, bias = self.w_output_gate["w_stm"], self.w_output_gate["w_in"], self.w_output_gate["bias"]
       
       percentage_ltm = sigmoid(self.stm * w_stm + self.input * w_in + bias)
       self.stm = percentage_ltm * tanh_activation(self.ltm) 

    def next(self, input):
       self.input = input
       self.forget_gate()
       self.input_gate()
       self.output_gate()
       return (round(self.ltm, 1), round(self.stm,1))

    
    def backpropagation():
        pass


if __name__ == "__main__":
    ltm, stm = (0,0)
    lstm = LSTM(ltm=ltm,stm=stm,input=0)
    data = [1,0.5,0.25,1]
    for i in range(len(data)):
        output = lstm.next(data[i])
        ltm, stm = output
        print(output)
        print("\n\nANOTHER EPOCH\n\n")
    print(ltm, stm)
