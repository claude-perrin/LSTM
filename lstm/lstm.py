import pandas as pd              
import matplotlib.pyplot as plt  
from activation_functions import *



class LSTM:
    def __init__(self):
        self.long_term_memory = 0
        self.short_term_memory = 0

        #Long-term to remember Forget Gate
        self.w_stm_ltm_to_remember = 2.7
        self.w_input_ltm_to_remember = 1.63
        self.bias_ltm_to_remember = 1.62
        
        #Poterntial Memory to remember ltm InputGate
        self.w_stm_potential_remember_ltm = 2
        self.w_input_potential_remember_ltm = 1.65
        self.bias_potential_remember_ltm = 0.62
        
        #Potential Long-Term memory InputGate
        self.w_stm_potential_ltm = 1.41
        self.w_input_potential_ltm = 0.94
        self.bias_potential_ltm = -0.32

        #Potential memory to remember stm OutputGate
        self.w_stm_potential_remember_stm = 4.38
        self.w_input_potential_remember_stm = -0.19
        self.bias_potential_remember_stm = 0.59
    
    
    def percentage_to_remember(self, stm, input, w_stm, w_in, bias):
        return sigmoid(stm * w_stm + input * w_in + bias)
    
    def potential_ltm(self, stm, input, w_stm, w_in, bias):
        return tanh_activation(stm * w_stm + input * w_in + bias)
    
    def new_ltm(self, forgeted_ltm, remember_percentage_ltm, potential_ltm):
        return forgeted_ltm + remember_percentage_ltm * potential_ltm

    def potential_stm(self, remember_percentage_stm, new_ltm):
        return remember_percentage_stm * tanh_activation(new_ltm)



    def initialize_layer(self, input, stm, ltm):
       # Step 1 ForgetGate
       ltm_to_remember = self.percentage_to_remember(stm, input, w_stm=self.w_stm_ltm_to_remember, w_in=self.w_input_ltm_to_remember, bias=self.bias_ltm_to_remember)
       
       print("ltm_to_remember", ltm_to_remember)
       forgeted_ltm = ltm_to_remember*ltm
       print("forgeted_ltm", forgeted_ltm) 
       # Step 2 InputGate
       remember_percentage_ltm = self.percentage_to_remember(stm, input, w_stm=self.w_stm_potential_remember_ltm, w_in=self.w_input_potential_remember_ltm, bias=self.bias_potential_remember_ltm)
       print("remember_percentage_ltm", remember_percentage_ltm)
       potential_ltm = self.potential_ltm(stm,input, w_stm=self.w_stm_potential_ltm, w_in=self.w_input_potential_ltm, bias=self.bias_potential_ltm)
       print("potential_ltm",potential_ltm)
       new_ltm = self.new_ltm(forgeted_ltm, remember_percentage_ltm, potential_ltm) 
       print("new_ltm", new_ltm)
       # Step 3 OutputGate
       remember_percentage_stm = self.percentage_to_remember(stm,input, w_stm=self.w_stm_potential_remember_stm, w_in=self.w_stm_potential_remember_stm, bias=self.bias_potential_remember_stm)
       print("remember_percentage_stm", remember_percentage_stm)
       new_stm = self.potential_stm(remember_percentage_stm, new_ltm)
       print("new_stm", new_stm)
       return (round(new_ltm, 1), round(new_stm,1))

    
    def backpropagation():
        pass


if __name__ == "__main__":
    lstm = LSTM()
    data = [1,0.5,0.25,1]
    ltm, stm = (0,0)
    for i in range(len(data)):
        output = lstm.initialize_layer(data[i],stm,ltm)
        ltm, stm = output
        print(output)
        print("\n\nANOTHER EPOCH\n\n")
    print(ltm, stm)
