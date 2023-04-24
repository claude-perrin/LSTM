import pandas as pd              
import matplotlib.pyplot as plt  
from activation_functions import *



class LSTM:
    def __init__(self, ltm, stm, input):
        self.input = input
        self.ltm = ltm
        self.stm = stm

        #Long-term to remember Forget Gate
        self.w_forget_gate = { "w_stm": 2.7, "w_in": 1.63, "bias": 1.62 }#       

        #Poterntial Memory to remember ltm InputGate
        self.w_input_gate_sigmoid = { "w_stm": 2, "w_in": 1.65, "bias": 0.62 } 
        #Potential Long-Term memory InputGate
        self.w_input_gate_tan = { "w_stm": 1.41, "w_in": 0.94, "bias": -0.32 }

        #Potential memory to remember stm OutputGate
        self.w_output_gate = { "w_stm": 4.38, "w_in": -0.19, "bias": 0.59 }

    def forget_gate(self):
        w_stm, w_in, bias = self.w_forget_gate["w_stm"], self.w_forget_gate["w_in"], self.w_forget_gate["bias"]
        forget_percentage = sigmoid(self.stm * w_stm + self.input * w_in + bias)
        return forget_percentage


    def input_gate(self): 
        w_stm_sig, w_in_sig, bias_sig = self.w_input_gate_sigmoid["w_stm"], self.w_input_gate_sigmoid["w_in"], self.w_input_gate_sigmoid["bias"]
        w_stm_tan, w_in_tan, bias_tan = self.w_input_gate_tan["w_stm"], self.w_input_gate_tan["w_in"], self.w_input_gate_tan["bias"]

        # i
        percentage_ltm = sigmoid(self.stm * w_stm_sig + self.input * w_in_sig + bias_sig)

        # g
        potential_ltm = tanh_activation(self.stm * w_stm_tan + self.input * w_in_tan + bias_tan)

        return percentage_ltm, potential_ltm


    def output_gate(self): 
        w_stm, w_in, bias = self.w_output_gate["w_stm"], self.w_output_gate["w_in"], self.w_output_gate["bias"]

        output_gate_out = sigmoid(self.stm * w_stm + self.input * w_in + bias)
        return output_gate_out

    def next(self, input):
        self.input = input
        forget_gate_out = self.forget_gate()

        percentage_ltm, potential_ltm = self.input_gate()
        input_gate_out = percentage_ltm*potential_ltm

        output_gate_out = self.output_gate()
       
        self.ltm = self.ltm * forget_gate_out + input_gate_out
        self.stm = output_gate_out * tanh_activation(self.ltm)
        return (round(self.ltm, 1), round(self.stm,1))
    
    
    def backpropagation(orig_val, pred_val, ltm, stm, output_gate_out,forget_gate_out, input, i, g):
        """
        orig_val is the one that is pointed by supervised learning
        pred_val is the value predicted by the lstm 
        ct = self.ltm
        ht = self.scm
        o = output_gate_out
        i = input_gate_out sigmoid
        g = input_gate_out tanh
        f = forget_gate_out
        x = input
        """

        E_delta = orig_val - pred_val
        
        gradients = {}
        # Gradient with respect to output gate
        gradients["dE_do"] = E_delta * tanh(ltm)

        #Gradient with respect to ltm
        gradients["dE_dltm"] = E_delta * output_gate_out * (1-tanh_activation(ltm)^2)

        #Gradient with respect to input gate dE/di, dE/dg
        gradients["dE_di"] = E_delta * output_gate_out * (1-tanh_activation(ltm)^2) * g
        gradients["dE_dg"] = E_delta * output_gate_out * (1-tanh_activation(ltm)^2) * i

        # Gradient with respect to forget gate

        gradients["dE_df"] = E_delta * output_gate_out * (1-tanh_activation(ltm)^2) * self.ltm
        
        # Gradient with respect to self.ltm

        gradients["dE_dself_ltm"] =  E_delta * output_gate_out * (1-tanh_activation(ltm)^2) * forget_gate_out

        # Gradient with respect to output gate weights

        gradients["dE_dw_X_output_gate"] =  gradients["dE_do"] * output_gate_out * (1-output_gate_out) * input
        gradients["dE_dw_stm_output_gate"] = gradients["dE_do"] * output_gate_out * (1-output_gate_out) * self.stm
        gradients["dE_db_bias_output_gate"] = gradients["dE_do"] * output_gate_out * (1-output_gate_out)
        
        # Gradient with respect to forget gate weights
        
        gradients["dE_dw_X_forget_gate"] =  gradients["dE_df"] * forget_gate_out * (1-forget_gate_out) * input
        gradients["dE_dw_stm_forget_gate"] = gradients["dE_df"] * forget_gate_out * (1-forget_gate_out) * self.stm
        gradients["dE_db_bias_forget_gate"] = gradients["dE_df"] * forget_gate_out * (1-forget_gate_out)
        
        # Gradient with respect to input gate weights:

        gradients["dE_dw_X_input_gate_sig"] =  gradients["dE_di"] * g * (1-g) * input
        gradients["dE_dw_stm_input_gate_sig"] = gradients["dE_di"] * g * (1-g) * self.stm
        gradients["dE_db_bias_input_gate_sig"] = gradients["dE_di"] * g * (1-g)


        gradients["dE_dw_X_input_gate_tan"] =  gradients["dE_dg"] * i  * (1-i) * input
        gradients["dE_dw_stm_input_gate_tan"] = gradients["dE_dg"] * i  * (1-i) * self.stm
        gradients["dE_db_bias_input_gate_tan"] = gradients["dE_dg"] * i  * (1-i)

        print(gradients)

    def update_weights(self, grads, learning_rate):
        #update output gate weights
        self.w_output_gate["w_stm"] -= learning_rate * grads["dE_dw_stm_output_gate"] 
        self.w_output_gate["w_in"] -= learning_rate * grads["dE_dw_X_output_gate"] 
        
        #update input gate sigmoid weights
        self.w_input_gate_sigmoid["w_stm"] -= learning_rate * grads["dE_dw_stm_input_gate"] 
        self.w_input_gate_signoid["w_in"] -= learning_rate * grads["dE_dw_X_input_gate"] 
            
        #update input gate tanh weights
        self.w_input_gate_tan["w_stm"] -= learning_rate * grads["dE_dw_stm_input_gate_tan"] 
        self.w_input_gate_tan["w_in"] -= learning_rate * grads["dE_dw_X_input_gate_tan"] 
        

        #update forget gate weights
        self.w_forget_gate["w_stm"] -= learning_rate * grads["dE_dw_stm_forget_gate"] 
        self.w_forget_gate["w_in"] -= learning_rate * grads["dE_dw_X_forget_gate"] 

    def get_weights(self):
        return {"forget_gate":self.w_forget_gate, "input_sig": self.w_input_gate_sigmoid, "input_tan": self.w_input_gate_tan, "output": self.w_output_gate}
    

if __name__ == "__main__":
    ltm, stm = (0,0)
    lstm = LSTM(ltm=ltm,stm=stm,input=0)
    data = [0,0.5,0.25,1]
#    for i in range(len(data)):
#        output = lstm.next(data[i])
#        ltm, stm = output
#        print(output)
#        print("\n\nANOTHER LAYER\n\n")
    print(ltm, stm)
    print(lstm.get_weights())
