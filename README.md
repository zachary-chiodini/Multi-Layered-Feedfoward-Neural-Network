<h1 style="color:#F4EE00; font-family:cambria"><i>Multi-Layered, Feedforward Neural Network</i></h1>

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>This is a multi-layred, feedforward neural network written from scratch in Python.</i>
</span>
<hr>

<h2 style="color:#F4EE00; font-family:cambria"><i>Importing Libraries</i></h2>
<hr>


```python
import numpy as np
from typing import Any, Callable, List
from nptyping import NDArray, Float64
```

<h2 style="color:#F4EE00; font-family:cambria"><i>Defining Data Structures</i></h2>
<hr>

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Feature Matrix</i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{blue}{X} = \begin{bmatrix}
    1 & \color{blue}{x_{1,1}} & \dots & \color{blue}{x_{1,n}} \\
    1 & \color{blue}{x_{2,1}} & \dots & \color{blue}{x_{2,n}} \\
    \vdots & \vdots & \ddots & \vdots \\
    1 & \color{blue}{x_{m,1}} & \dots & \color{blue}{x_{m,n}} \\
\end{bmatrix} = \begin{bmatrix}
    \color{blue}{x_{1,0}} & \color{blue}{x_{1,1}} & \dots & \color{blue}{x_{1,n}} \\
    \color{blue}{x_{2,0}} & \color{blue}{x_{2,1}} & \dots & \color{blue}{x_{2,n}} \\
    \vdots & \vdots & \ddots & \vdots \\
    \color{blue}{x_{m,0}} & \color{blue}{x_{m,1}} & \dots & \color{blue}{x_{m,n}} \\
\end{bmatrix}
$</span>


```python
Number_of_Examples = Any
Number_of_Features_Plus_One = Any
Feature_Matrix = NDArray[ ( Number_of_Examples, Number_of_Features_Plus_One ), Float64 ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Target Matrix</i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{yellow}{Y} = \begin{bmatrix}
    \color{yellow}{y_{1,1}} & \color{yellow}{y_{1,2}} & \dots & \color{yellow}{y_{1,n}} \\
    \color{yellow}{y_{2,1}} & \color{yellow}{y_{2,2}} & \dots & \color{yellow}{y_{2,n}} \\
    \vdots & \vdots & \ddots & \vdots \\
    \color{yellow}{y_{m,1}} & \color{yellow}{y_{m,2}} & \dots & \color{yellow}{y_{m,n}}
\end{bmatrix}
$</span>


```python
Number_of_Target_Variables = Any
Target_Matrix = NDArray[ ( Number_of_Examples, Number_of_Target_Variables ), Float64 ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Output</i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{yellow}{\hat{Y}} = \begin{bmatrix}
    \color{yellow}{\hat{y}_{1,1}} & \color{yellow}{\hat{y}_{1,2}} & \dots & \color{yellow}{\hat{y}_{1,n}} \\
    \color{yellow}{\hat{y}_{2,1}} & \color{yellow}{\hat{y}_{2,2}} & \dots & \color{yellow}{\hat{y}_{2,n}} \\
    \vdots & \vdots & \ddots & \vdots \\
    \color{yellow}{\hat{y}_{m,1}} & \color{yellow}{\hat{y}_{m,2}} & \dots & \color{yellow}{\hat{y}_{m,n}}
\end{bmatrix}
$</span>


```python
Output_Matrix = NDArray[ ( Number_of_Examples, Number_of_Target_Variables ), Float64 ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Perceptron</i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{green}{P}^{(ℓ)}_{\color{green}{k}} = \begin{bmatrix}
    \color{green}{\omega_{0,k}} \\
    \color{green}{\omega_{1,k}} \\
    \vdots \\
    \color{green}{\omega_{j,k}}
\end{bmatrix}
$</span>


```python
Weight = Float64
Number_of_Inputs_Plus_One = Any
Perceptron = NDArray[ ( Number_of_Inputs_Plus_One, 1 ), Weight ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Layer</i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{green}{\Omega}^{(ℓ)} = \begin{bmatrix}
    \color{green}{P}^{(ℓ)}_{\color{green}{1}} & 
    \color{green}{P}^{(ℓ)}_{\color{green}{2}} & 
    \dots & 
    \color{green}{P}^{(ℓ)}_{\color{green}{k}}
\end{bmatrix} = \begin{bmatrix}
    \color{green}{\omega_{0,1}} & \color{green}{\omega_{0,2}} & \dots & \color{green}{\omega_{0,k}} \\
    \color{green}{\omega_{1,1}} & \color{green}{\omega_{1,2}} & \dots & \color{green}{\omega_{1,k}} \\
    \vdots & \vdots & \ddots & \vdots \\
    \color{green}{\omega_{j,1}} & \color{green}{\omega_{j,2}} & \dots & \color{green}{\omega_{j,k}}
\end{bmatrix}
$</span>


```python
Number_of_Perceptrons = Any
Layer = NDArray[ ( Number_of_Inputs_Plus_One, Number_of_Perceptrons ), Perceptron ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Neural Network</i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{green}{N}=\begin{bmatrix}
   \color{green}{\Omega}^{(1)} &
   \color{green}{\Omega}^{(2)} & 
   \dots & 
   \color{green}{\Omega}^{(ℓ)}
\end{bmatrix}
$</span>


```python
Network = List[ Layer ]
```

<h2 style="color:#F4EE00; font-family:cambria"><i>Creating the Neural Network</i></h2>
<hr>

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Initialization</i>
</span>


```python
class FeedForwardNeuralNetwork :
    
    def __init__( self,
        hidden_layers : List[ int ] = [],
        funk : Callable[ [ Float64 ], Float64 ] \
                 = lambda x : 1 / ( 1 + np.exp( -x ) ),
        cost : Callable[ [ NDArray[ Float64 ] ], Float64 ] \
                 = lambda y1, y2 : np.square( y1 - y2 ).sum()
        ) -> None :
        self.funk = funk
        self.cost = cost
        self.network : Network = []
        self.input_layer : List[ str ] = []
        self.hidden_layers = hidden_layers
        return
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Forward Propagation</i>
</span>
<br>
<br>
<table align="left">
    <tr>
        <td style="font-size:14px">$\color{blue}{Output Layer}$</td>
        <td style="font-size:24px">$\color{blue}{L}^{(ℓ)}=
            \color{red}{f}⊙(\color{blue}{L}^{(ℓ-1)}\color{green}{\Omega}^{ℓ})$</td>
    </tr>
    <tr>
        <td style="font-size:14px">$\color{blue}{Hidden Layers}$</td>
        <td style="font-size:24px">
            $\color{blue}{L}^{(ℓ-1)}=\begin{bmatrix}
                    1 & \color{red}{f}⊙(\color{blue}{L}^{(ℓ-2)}\color{green}{\Omega}^{ℓ})
                \end{bmatrix}$
        </td>
    </tr>
    <tr>
        <td style="font-size:14px">$\color{blue}{Input Layer}$</td>
        <td style="font-size:24px">$\color{blue}{L}^{(0)} = \color{blue}{X}$</td>
    </tr>
</table>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def forwardpropagate( self, X : Feature_Matrix ) -> Output_Matrix :
        L = X
        for i in range( len( self.network ) - 1 ) :
            L = np.concatenate(
                np.ones( len( X ), 1 ),
                self.funk( np.matmul( L, self.network[ i ] ) ),
                axis = 1
                )
        return self.funk( np.matmul( L, self.network[ -1 ] ) )
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Training Algorithm (Backpropagation)</i>
</span>
<br>
<br>
<table align="left">
    <tr>
        <td style="font-size:14px">$\color{blue}{Output Layer}$</td>
        <td style="font-size:24px">$
        \nabla^{(ℓ)}_{\color{pink}{a}}\color{pink}{C}(\color{yellow}{\hat{Y}};\color{yellow}{Y}) = 
        \nabla^{(ℓ)}_{\color{pink}{a}}\color{pink}{C}(\color{yellow}{\hat{Y}};\color{pink}{a}^{(ℓ)}_{\color{pink}{1}}, \color{pink}{a}^{(ℓ)}_{\color{pink}{2}},...,\color{pink}{a}^{(ℓ)}_{\color{pink}{k}}) = \\
            \begin{bmatrix}
                \frac{\partial \color{pink}{C}(\color{yellow}{\hat{Y}};\color{pink}{a}^{(ℓ)}_{\color{pink}{1}}, \color{pink}{a}^{(ℓ)}_{\color{pink}{2}},...,\color{pink}{a}^{(ℓ)}_{\color{pink}{k}} ))}{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{1}}} &
                \frac{\partial \color{pink}{C}(\color{yellow}{\hat{Y}};\color{pink}{a}^{(ℓ)}_{\color{pink}{1}}, \color{pink}{a}^{(ℓ)}_{\color{pink}{2}},...,\color{pink}{a}^{(ℓ)}_{\color{pink}{k}} )) }{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{2}}} &
                \dots & 
                \frac{\partial \color{pink}{C}(\color{yellow}{\hat{Y}};\color{pink}{a}^{(ℓ)}_{\color{pink}{1}}, \color{pink}{a}^{(ℓ)}_{\color{pink}{2}},...,\color{pink}{a}^{(ℓ)}_{\color{pink}{k}} )) }{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{k}}} 
            \end{bmatrix}
        $</td>
    </tr>
</table>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def train( self, X : Feature_Matrix, Y : Target_Matrix ) :
        output : Output_Matrix = self.forwardpropagate( X )
        grad_cost = np.gradient( self.cost( output, Y ) )
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Training Algorithm (Backpropagation)</i>
</span>
<br>
<br>
<table align="left">
    <tr>
        <td style="font-size:14px">$\color{blue}{Output Layer}$</td>
        <td style="font-size:24px">$
        \nabla^{(ℓ)}_{\color{orange}{z}}\color{pink}{C}=
        (\color{pink}{f}'⊙\color{orange}{Z}[ℓ-1])⊙\nabla^{(ℓ)}_{\color{pink}{a}}\color{pink}{C}⊙\mathbf{1}_{j\times k} \\
        \nabla^{(ℓ)}_{\color{green}{\Omega}}\color{pink}{C}=
        (\color{pink}{A}[ℓ-1])^{T}⊙\nabla^{(ℓ)}_{\color{orange}{z}}\color{pink}{C} \\
        \color{green}{N}[ℓ-1] = \color{green}{N}[ℓ-1] - r\nabla^{(ℓ)}_{\color{green}{\Omega}}\color{pink}{C}
        $</td>
    </tr>
    <tr>
        <td style="font-size:14px">$\color{blue}{Containers}$</td>
        <td style="font-size:24px">
            $
            \color{pink}{A}[ℓ-1] = \color{blue}{L}^{(ℓ)}\\
            \color{orange}{Z}[ℓ-1] = \color{blue}{L}^{(ℓ-1)}\color{green}{\Omega}^{(ℓ)}
            $
        </td>
    </tr>
</table>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def __forwardpropagate( self, X : Feature_Matrix, A, Z ) -> Output :
        L = X
        A.append( L )
        for i in range( len( self.network ) - 1 ) :
            product = np.matmul( L, self.network[ i ] )
            L = np.concatenate(
                np.ones( len( X ), 1 ),
                self.funk( product ),
                axis = 1
                )
            Z.append( product )
            A.append( L )
        product = np.matmul( L, self.network[ -1 ] ) 
        output : Output = self.funk( product )
        Z.append( product )
        A.append( output )
        return output
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Training Algorithm (Backpropagation)</i>
</span>
<br>
<br>
<table align="left">
    <tr>
        <td style="font-size:14px">$\color{blue}{Output Layer}$</td>
        <td style="font-size:24px">$
        \nabla^{(ℓ)}_{\color{orange}{z}}\color{pink}{C}=
        (\color{pink}{f}'⊙\color{orange}{Z}[ℓ-1])⊙\nabla^{(ℓ)}_{\color{pink}{a}}\color{pink}{C}⊙\mathbf{1}_{j\times k} \\
        \nabla^{(ℓ)}_{\color{green}{\Omega}}\color{pink}{C}=
        (\color{pink}{A}[ℓ-1])^{T}⊙\nabla^{(ℓ)}_{\color{orange}{z}}\color{pink}{C} \\
        \color{green}{N}[ℓ-1] = \color{green}{N}[ℓ-1] - r\nabla^{(ℓ)}_{\color{green}{\Omega}}\color{pink}{C}
        $</td>
    </tr>
</table>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def train( self, 
        X : Feature_Matrix, 
        Y : Target_Matrix, 
        r : float, # learning rate
        h : float, # convergence
        iters = 100 
        ) -> None :
        A, Z = [], []
        grad_cost = np.inf
        while abs( grad_cost ) > h :
            output : Output_Matrix = self.__forwardpropagate( X, A, Z )
            grad_cost = np.gradient( self.cost( output, Y ) )
            
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Training Algorithm (Backpropagation)</i>
</span>
<br>
<br>
<table align="left">
    <tr>
        <td style="font-size:14px">$\color{blue}{Hidden Layers}$</td>
        <td style="font-size:24px">
            $
            \nabla^{(i)}_{\color{orange}{z}}\color{pink}{C}=
            (\color{pink}{f}'⊙\color{orange}{Z}[i])⊙\color{green}{\Omega}^{(i+1)}(\nabla^{(i+1)}_{\color{orange}{z}}\color{pink}{C})^{T} \\
            \nabla^{(ℓ)}_{\color{green}{\Omega}}\color{pink}{C}=
            (\color{pink}{A}[i])^{T}⊙\nabla^{(i)}_{\color{orange}{z}}\color{pink}{C} \\
            \color{green}{N}[i] = \color{green}{N}[i] - r\nabla^{(i)}_{\color{green}{\Omega}}\color{pink}{C}
            $
        </td>
    </tr>
</table>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def train( self ) :
        pass
```
