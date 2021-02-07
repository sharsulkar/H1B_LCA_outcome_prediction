# LSTM: Long Short Term Memory

## From RNN to LSTM

### Gated architecture

As seen in <insert back reference to RNN chapter section "Ways to fix RNN issues" here>, simple RNN cell architecture has limitations in learning long term dependencies from input sequence due to vanishing/exploding gradients. In 1997, a better architecture called Long Short-Term Memory (LSTM) was proposed  with the central idea of "Constant Error Carousal" which enforces constant (thus neither exploding or vanishing) error flow through internal states of special self-connected units [1]. These "special self-connected units" are also called *memory cells* because the cell state is decoupled from the output and hidden states connected and thus can be made to persist its state by using special gates (similar to a gated architecture of a transistor).

### Forward Pass

The diagrams and descriptions from [2] have been used with some modifications for the forward pass of LSTM as they provide a simple, intuitive description.

As previously stated, the central idea of LSTM is its ability to pass on the error between its states without much change. As can be seen in Figure 1 below, the cell state *C~t~*  carries the previous state to the next and error from next to previous state does not have many transformations. It is like a "carousal" transporting contents without modification, except as applied by the two linear interactions. 

![](/images/LSTM/LSTM Constant error carousal.png)

*Figure 1 : Constant Error Carousal in LSTM*

To update the cell state, two operations can be performed, namely, forgetting (partially or fully) the stored state and appending new information to what is stored. During training, the LSTM learns to decide which operation to apply for what input (external input X and previous state H), by learning parameters controlling these operations. 

The **forget gate**, as shown in the Figure 2 below, which is basically a sigmoid operation on the input signal (X,H) which scales the previous state between {0,1} where 0 is forget all, 1 is no change and partial update for all values in between.

![](/images/LSTM/LSTM Forget gate.png)

​																									*Figure 2 : Forget gate in LSTM*

As a side note, the original LSTM paper [1] did not mention a forget gate (or peephole connections), which was added to the architecture in 2000 by [3].

The **input gate and Candidate memory**, as shown in Figure 3 below, decides what new information coming in from input (X,H) should be stored in the current state. It is a 2 step process. 

1.  A sigmoid layer called **input gate layer** decides which values should be updated
2. The tanh layer called **Candidate memory** creates a vector of candidate updates that will be added to the previous cell state.

![](/images/LSTM/LSTM input gate.png)

​																									*Figure 3 : Input gate and Candidate memory in LSTM*

Next, we apply the forget and update operations calculated till now to the previous cell state to get the new cell state, as shown in Figure 4 below.

![](/images/LSTM/LSTM update cell state.png)

​																									*Figure 4 : Updating cell state in LSTM*

Last, we compute the output of the LSTM cell which is also gated using an **output gate layer** as shown in Figure 5 below. First we decide to which part of the input (X,H) we are going to pass on to the output. Then we pass the cell state through a tanh and multiply it with the output of sigmoid layer so we only output the parts that are decided.

![](/images/LSTM/LSTM output gate.png)

​																									*Figure 5 : Output gate in LSTM*

Mathematically, these operations can be described by the equations below -
$$
f_t=\sigma(W_f.[h_{t-1},x_t]+b_f)
$$

$$
i_t=\sigma(W_i.[h_{t-1},x_t]+b_i)
$$

$$
\tilde{C}_t=tanh(W_c.[h_{t-1},x_t]+b_c)
$$

$$
C_t=f_t*C_{t-1}+i_t*\tilde{C}_t
$$

$$
o_t=\sigma(W_o.[h_{t-1},x_t]+b_o)
$$

$$
h_t=o_t*tanh(C_t)
$$

where the weights $$[W_f,W_i,W_c,W_o]$$ and biases $$[b_f,b_i,b_c,b_o]$$ are the trainable parameters of LSTM.

### Backward Pass through Time

 Again, the diagram and equations used in [4] are heavily referred here. Although the notations used between forward and backward passes are different, the diagrams are intuitive and can be used to easily sync them.

![](/images/LSTM/LSTM backpropogation.png)

​																									*Figure 6 : Back Propagation in LSTM*

Similar to RNN, lets assume the loss function J is cross-entropy. Refer equations 6,7 and 8 in RNN article for more details.

**Backpropagation equations :**

**Output:**
$$
\frac{\partial J}{\partial v_t}=o_t-y_t
$$

$$
\frac{\partial J}{\partial W_v}=\frac{\partial J}{\partial v_t}\frac{\partial v_t}{\partial W_v}=(o_t-v_t).h_t^T
$$

$$
\frac{\partial J}{\partial b_v}=\frac{\partial J}{\partial v_t}\frac{\partial v_t}{\partial b_v}=(o_t-v_t)
$$



Note that Equation 8 and 9 are similar to RNN.

**Hidden state:**
$$
\frac{\partial J}{\partial h_t}=\frac{\partial J}{\partial v_t}\frac{\partial v_t}{\partial h_t}=W_v^T(o_t-v_t)
$$
**Output gate**
$$
\frac{\partial J}{\partial o_t}=\frac{\partial J}{\partial h_t}\frac{\partial h_t}{\partial o_t}=\frac{\partial J}{\partial h_t}\bigodot tanh(c_t)
$$

$$
\frac{\partial J}{\partial a_o}=\frac{\partial J}{\partial o_t}\frac{\partial o_t}{\partial a_o}=\frac{\partial J}{\partial h_t}\bigodot tanh(c_t)\bigodot o_t(1-o_t)
$$

$$
\frac{\partial J}{\partial W_o}=\frac{\partial J}{\partial a_o}\frac{\partial a_o}{\partial W_o}=\frac{\partial J}{\partial a_o}.Z_t^T
$$

$$
\frac{\partial J}{\partial b_o}=\frac{\partial J}{\partial a_o}\frac{\partial a_o}{\partial b_o}=\frac{\partial J}{\partial a_o}
$$

**Cell state**
$$
\frac{\partial J}{\partial c_t}=\frac{\partial J}{\partial h_t}\frac{\partial h_t}{\partial c_t}=\frac{\partial J}{\partial h_t}\bigodot o_t \bigodot(1- tanh(c_t)^2)
$$

$$
\frac{\partial J}{\partial \hat{c}_t} = \frac{\partial J}{\partial c_{t}}\frac{\partial c}{\partial \hat{c}_t}=\frac{\partial J}{\partial c_{t}} \odot i_{t}
$$

$$
\frac{\partial J}{\partial a_c}=\frac{\partial J}{\partial \hat c_{t}}\frac{\partial \hat c_{t}}{\partial a_c}=\frac{\partial J}{\partial c_t} \bigodot i_t \bigodot (1-\hat c_t^2)
$$

$$
\frac{\partial J}{\partial W_c}=\frac{\partial J}{\partial a_c}\frac{\partial a_c}{\partial W_c}=\frac{\partial J}{\partial a_c}.z_t^T
$$

$$
\frac{\partial J}{\partial b_c}=\frac{\partial J}{\partial a_c}\frac{\partial a_c}{\partial b_c}=\frac{\partial J}{\partial a_c}
$$



**Input gate**
$$
\frac{\partial J}{\partial i_t}=\frac{\partial J}{\partial c_t}\frac{\partial c_t}{\partial i_t}=\frac{\partial J}{\partial c_t}\bigodot \hat c_t
$$

$$
\frac{\partial J}{\partial a_i}=\frac{\partial J}{\partial i_{t}}\frac{\partial i_{t}}{\partial a_i}=\frac{\partial J}{\partial c_t} \bigodot \hat c_t \bigodot i_t(1-i_t)
$$

$$
\frac{\partial J}{\partial W_i}=\frac{\partial J}{\partial a_i}\frac{\partial a_i}{\partial W_i}=\frac{\partial J}{\partial a_i}.z_t^T
$$

$$
\frac{\partial J}{\partial b_i}=\frac{\partial J}{\partial a_i}\frac{\partial a_i}{\partial b_i}=\frac{\partial J}{\partial a_i}
$$

**Forget gate**
$$
\frac{\partial J}{\partial f_t}=\frac{\partial J}{\partial c_t}\frac{\partial c_t}{\partial f_t}=\frac{\partial J}{\partial c_t}\bigodot c_{t-1}
$$

$$
\frac{\partial J}{\partial a_f}=\frac{\partial J}{\partial f_t}\frac{\partial f_t}{\partial a_f}=\frac{\partial J}{\partial c_t} \bigodot c_{t-1} \bigodot f_t(1-f_t)
$$

$$
\frac{\partial J}{\partial W_f}=\frac{\partial J}{\partial a_f}\frac{\partial a_f}{\partial W_f}=\frac{\partial J}{\partial a_f}.z_t^T
$$

$$
\frac{\partial J}{\partial b_f}=\frac{\partial J}{\partial a_f}\frac{\partial a_f}{\partial b_f}=\frac{\partial J}{\partial a_f}
$$

**Input**
$$
\frac{\partial J}{\partial z_{t}} = \frac{\partial J}{\partial a_{f}} \cdot \frac{\partial a_{f}}{\partial z_{t}} + \frac{\partial J}{\partial a_{i}} \cdot \frac{\partial a_{i}}{\partial z_{t}} + \frac{\partial J}{\partial a_{o}} \cdot \frac{\partial a_{o}}{\partial z_{t}} + \frac{\partial J}{\partial a_{c}} \cdot \frac{\partial a_{c}}{\partial z_{t}}
$$

$$
=  W_{f}^T \cdot \frac{\partial J}{\partial a_{f}} +W_{i}^T \cdot \frac{\partial J}{\partial a_{i}} + W_{o}^T \cdot \frac{\partial J}{\partial a_{o}} + W_{c}^T \cdot \frac{\partial J}{\partial a_{c}}
$$

$$
\frac{\partial J}{\partial h_{t-1}} = \frac{\partial J}{\partial z_{t}}[:n_{h}, :]
$$

$$
\frac{\partial J}{\partial c_{t-1}} = \frac{\partial J}{\partial c_{t}} \cdot \frac{\partial c_{t}}{\partial c_{t-1}} = \frac{\partial J}{\partial c_{t}} \odot f_{t}
$$

These equations for forward and backward pass are computed T times in each iteration (once for each time step $$t \epsilon T$$). At the end of each iteration, the weights are updated using the accumulated loss gradient wrt to each parameter. Equations are as given below -
$$
\frac{\partial J}{\partial W_{f}} = \sum\limits_{t}^T \frac{\partial J}{\partial W_{f}^t}, \mspace{31mu} W_{f} \pm \alpha * \frac{\partial J}{\partial W_{f}}
$$

$$
\frac{\partial J}{\partial W_{i}} = \sum\limits_{t}^T \frac{\partial J}{\partial W_{i}^t}, \mspace{31mu} W_{i} \pm \alpha * \frac{\partial J}{\partial W_{i}}
$$

$$
\frac{\partial J}{\partial W_{o}} = \sum\limits_{t}^T \frac{\partial J}{\partial W_{o}^t}, \mspace{31mu} W_{o} \pm \alpha * \frac{\partial J}{\partial W_{o}}
$$

$$
\frac{\partial J}{\partial W_{c}} = \sum\limits_{t}^T \frac{\partial J}{\partial W_{c}^t}, \mspace{31mu} W_{c} \pm \alpha * \frac{\partial J}{\partial W_{c}}
$$

$$
\frac{\partial J}{\partial W_{v}} = \sum\limits_{t}^T \frac{\partial J}{\partial W_{v}^t}, \mspace{31mu} W_{v} \pm \alpha * \frac{\partial J}{\partial W_{v}}
$$

Similar equations apply for bias parameter updates.



## How LSTM solves vanishing/exploding gradient problem

As seen in the RNN article equation 17, the gradients in RNN are unstable because there is iterative matrix multiplication with W~h~ where the matrix may have small or large values. In LSTM, differentiating equation 4 wrt C~t~ from the current time period t to a future time period t', we get 
$$
\frac{\partial C_{t'}}{\partial C_t} = \prod_{k=1}^{t' - t} f_{t+k}
$$
We also derived a similar equation for $$\frac{\partial J}{\partial c_{t-1}} $$ in equation 31. Both of there show that unlike RNN, LSTM does not have any iterative matrix multiplication that can cause gradients and error to shrink or swell when being passed on between the cell states. One more observation is that the forget gate controls how much of the error and thus gradient will be used to update the previous state. If the forget date is closed, f~t~ = 0 and no error is passed to previous states thus preserving the cell state indefinitely. Conversely, when f~t~=1, the gradient does not decay and is passed to previous states in its entirety. This is what is referred to as the "Constant Error Carousal" in the original LSTM paper [1] (Note that as forget gate was not part of the original paper, f~t~ was not available to control the error flow).

## Key LSTM variants

### Peepholes

Adding peephole connections to all gates from the previous cell state was suggested in the same paper [3] that introduced forget gates to LSTM. 

![](/images/LSTM/LSTM peephole connection.png)

​																									*Figure 7 : Peephole connections in LSTM*

Equations 1,2 and 5 above will become 
$$
f_t=\sigma(W_f.[C_{t-1},h_{t-1},x_t]+b_f)
$$

$$
i_t=\sigma(W_i.[C_{t-1},h_{t-1},x_t]+b_i)
$$

$$
o_t=\sigma(W_o.[C_t,h_{t-1},x_t]+b_o)
$$

### Coupled Forget and Input Gates

This modification will only allow the architecture to forget the cell state when there is a corresponding input available. Conversely, the cell only inputs new values when it forgets something from the cell state. For this, in all above equations, we replace i~t~ with (1-f~t~). Specifically, equation 4 above will become 
$$
C_t=f_t*C_{t-1}+(1-f_t)*\tilde{C}_t
$$
![](/images/LSTM/LSTM coupled gates.png)

​													                                           *Figure 8 : Coupling forget and input gates in LSTM*

## Gated Recurrent Unit (GRU)

Gated Recurrent Unit was introduced in 2014 by J Chung et all [6] as a hidden unit in their proposed RNN encoder-decoder architecture.

![](/images/LSTM/GRU.png)

​																							Figure 9 : Gated Recurrent Unit architecture

As the name suggests, GRU has two types of gates - Reset Gate and Update gate.

The Reset gate acts similar to forget gate in LSTM and is used to wipe out some or all of the cell state. Mathematically it is described in below equation
$$
r_t=\sigma (W_r[h_{t-1},x_t]+b_r)
$$
The update gate, similar to input gate in LSTM,  computes the expected update from external inputs. 
$$
z_t=\sigma (W_z[h_{t-1},x_t]+b_z)
$$
The computation of the candidate hidden state is unique to GRU and is not similar to that of LSTM. Candidate hidden state is calculated as below
$$
\tilde h_t=tanh(W_x[x_t]+W_h(r_t \bigodot h_{t-1})+b_h)
$$

If the reset gate is close to 1, the candidate state will be computed by an RNN (notice the similarity with RNN hidden state equation) while when reset gate is close to 0, the prior state is defaulted and the candidate state is computed from scratch using a MLP.

Finally, the below equation gives how to update the cell state based on candidate state.
$$
h_t=(1-z_t) \bigodot h_{t-1}+z_t \bigodot \tilde h_t
$$
So basically the update gate decides how much to keep the previous state and how much to add the candidate state.

The output is usually a softmax on the updated cell state as given below
$$
Y_t=Softmax(W_oh_t + b_o)
$$



Although GRU is a gated architecture built over RNN and not necessarily derived from LSTM, it has many similarities and differences to LSTM.

#### Similarities with LSTM 

1. Gated architecture
2. Constant error flow path

#### Differences from LSTM

1. No separate class state, or a output gate making the hidden state always available to the output.
2. In LSTM, when calculating the candidate cell state $$\tilde{C}_t$$, the input gate does not consider the previous cell state $$C_{t-1}$$, rather it independently controls the update to cell state (through the update gate). In GRU, the update gate considers the previous state $$h_{t-1}$$ as input while computing the candidate cell state $$\tilde h_t$$ but does not independently control the amount of candidate cell state added to cell state.

## Code Implementation of LSTM

### From Scratch with TensorFlow

### Using Keras API



## References

[1]: https://www.bioinf.jku.at/publications/older/2604.pdf	"LSTM paper"
[2]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/	"Christopher Olah blog on LSTM"
[3]: http://www.felixgers.de/papers/phd.pdf	"LongShort-Term Memory in Recurrent Neural Networks - Felix Gers"
[4]: https://christinakouridi.blog/2019/06/19/backpropagation-lstm/	"LSTM backpropagation"
[5]: https://stats.stackexchange.com/questions/185639/how-does-lstm-prevent-the-vanishing-gradient-problem	"Stackexchange discussion on Vanishing gradient problem"

[6]: https://arxiv.org/pdf/1406.1078.pdf	"GRU paper"
[7]: https://arxiv.org/pdf/1412.3555.pdf



## Further Reading

In addition to the references, below resources are recommended for further reading on this topic -

https://d-nb.info/1082034037/34 - RNN, LSTM

http://arunmallya.github.io/writeups/nn/lstm/index.html#/ - good derivations

https://arxiv.org/pdf/1610.02583.pdf - Good content

https://d2l.ai/chapter_recurrent-modern/deep-rnn.html - good ebook

http://karpathy.github.io/2015/05/21/rnn-effectiveness/ - Fun project ideas

https://arxiv.org/pdf/1503.04069.pdf - LSTM survey

https://arxiv.org/pdf/1808.03314.pdf - Mathematical derivations

