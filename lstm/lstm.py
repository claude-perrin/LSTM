#import pandas as pd
#import matplotlib.pyplot as plt
from activation_functions import tanh_activation, sigmoid


class LSTM:
    def __init__(self):
        self.ltm = 0
        self.stm = 0

        # Long-term to remember Forget Gate
        self.w_forget_gate = { "w_stm": 2.7, "w_in": 1.63, "bias": 1.62 }

        # Poterntial Memory to remember ltm InputGate
        self.w_input_gate_sigmoid = { "w_stm": 2, "w_in": 1.65, "bias": 0.62 } 
        # Potential Long-Term memory InputGate
        self.w_input_gate_tan = { "w_stm": 1.41, "w_in": 0.94, "bias": -0.32 }

        # Potential memory to remember stm OutputGate
        self.w_output_gate = { "w_stm": 4.38, "w_in": -0.19, "bias": 0.59 }

    def forget_gate(self, input):
        w_stm, w_in, bias = self.w_forget_gate["w_stm"], self.w_forget_gate["w_in"], self.w_forget_gate["bias"]
        forget_percentage = sigmoid(self.stm * w_stm + input * w_in + bias)
        return forget_percentage

    def input_gate(self,input):
        w_stm_sig, w_in_sig, bias_sig = self.w_input_gate_sigmoid["w_stm"], self.w_input_gate_sigmoid["w_in"], self.w_input_gate_sigmoid["bias"]
        w_stm_tan, w_in_tan, bias_tan = self.w_input_gate_tan["w_stm"], self.w_input_gate_tan["w_in"], self.w_input_gate_tan["bias"]

        percentage_ltm = sigmoid(self.stm * w_stm_sig + input * w_in_sig + bias_sig)

        potential_ltm = tanh_activation(self.stm * w_stm_tan + input * w_in_tan + bias_tan)

        return percentage_ltm, potential_ltm

    def output_gate(self,input): 
        w_stm, w_in, bias = self.w_output_gate["w_stm"], self.w_output_gate["w_in"], self.w_output_gate["bias"]

        output_gate_out = sigmoid(self.stm * w_stm + input * w_in + bias)
        return output_gate_out

    def next(self, input):
        forget_gate_out = self.forget_gate(input)

        percentage_ltm, potential_ltm = self.input_gate(input)
        input_gate_out = percentage_ltm * potential_ltm

        output_gate_out = self.output_gate(input)

        self.ltm = self.ltm * forget_gate_out + input_gate_out
        self.stm = output_gate_out * tanh_activation(self.ltm)
        return (round(self.ltm, 1), round(self.stm,1))


    def backpropagation(self, orig_val, pred_val, ltm, stm, output_gate_out,forget_gate_out, input, i, g):
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
        dE_do = E_delta * tanh_activation(ltm)


        # Gradient with respect to input gate dE/di, dE/dg
        dE_di = E_delta * output_gate_out * (1 - tanh_activation(ltm)^2) * g
        dE_dg = E_delta * output_gate_out * (1 - tanh_activation(ltm)^2) * i

        # Gradient with respect to forget gate

        dE_df = E_delta * output_gate_out * (1 - tanh_activation(ltm)^2) * self.ltm

        # Gradient with respect to ltm

        gradients["dE_dltm"] = E_delta * output_gate_out * (1 - tanh_activation(ltm)^2)

        # Gradient with respect to self.ltm

        gradients["dE_dself_ltm"] = E_delta * output_gate_out * (1 - tanh_activation(ltm)^2) * forget_gate_out

        # Gradient with respect to output gate weights

        gradients["in_output_gate"] = dE_do * output_gate_out * (1 - output_gate_out) * input
        gradients["stm_output_gate"] = dE_do * output_gate_out * (1 - output_gate_out) * self.stm
        gradients["bias_output_gate"] = dE_do * output_gate_out * (1 - output_gate_out)

        # Gradient with respect to forget gate weights

        gradients["in_forget_gate"] = dE_df * forget_gate_out * (1 - forget_gate_out) * input
        gradients["stm_forget_gate"] = dE_df * forget_gate_out * (1 - forget_gate_out) * self.stm
        gradients["bias_forget_gate"] = dE_df * forget_gate_out * (1 - forget_gate_out)

        # Gradient with respect to input gate weights:

        gradients["in_input_gate_sig"] = dE_di * g * (1 - g) * input
        gradients["stm_input_gate_sig"] = dE_di * g * (1 - g) * self.stm
        gradients["bias_input_gate_sig"] = dE_di * g * (1 - g)


        gradients["in_input_gate_tan"] = dE_dg * i * (1 - i) * input
        gradients["stm_input_gate_tan"] = dE_dg * i * (1 - i) * self.stm
        gradients["bias_input_gate_tan"] = dE_dg * i * (1 - i)

        return gradients

    def update_weights(self, grads, learning_rate):
        #update output gate weights
        self.w_output_gate["w_stm"] -= learning_rate * grads["stm_output_gate"] 
        self.w_output_gate["w_in"] -= learning_rate * grads["in_output_gate"] 

        #update input gate sigmoid weights
        self.w_input_gate_sigmoid["w_stm"] -= learning_rate * grads["stm_input_gate"] 
        self.w_input_gate_signoid["w_in"] -= learning_rate * grads["in_input_gate"] 

        #update input gate tanh weights
        self.w_input_gate_tan["w_stm"] -= learning_rate * grads["stm_input_gate_tan"] 
        self.w_input_gate_tan["w_in"] -= learning_rate * grads["in_input_gate_tan"] 


        #update forget gate weights
        self.w_forget_gate["w_stm"] -= learning_rate * grads["stm_forget_gate"] 
        self.w_forget_gate["w_in"] -= learning_rate * grads["in_forget_gate"] 

    
    def train(self, input_array, target_array):
        for input in input_array:
            self.next(input)





    def get_weights(self):
        return {"forget_gate":self.w_forget_gate, "input_sig": self.w_input_gate_sigmoid, "input_tan": self.w_input_gate_tan, "output": self.w_output_gate}
    
if __name__ == "__main__":
    ltm, stm = (0,0)
    lstm = LSTM()
    data = [0,0.5,0.25,1]
    for i in range(len(data)):
        output = lstm.next(data[i])
        ltm, stm = output
        print(output)
        print("\n\nANOTHER LAYER\n\n")
print(ltm, stm)
for key,value in lstm.get_weights().items():
    print(key, value)
