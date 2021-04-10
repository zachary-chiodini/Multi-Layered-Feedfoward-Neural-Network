<h1 style="font-size:32px; color:#F4EE00; font-family:cambria"><i>Multi-Layered, Feedforward Neural Network</i></h1>

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>This is a multi-layred, feedforward neural network written from scratch in Python.</i>
</span>
<hr>

<h2 style="font-size:32px; color:#F4EE00; font-family:cambria"><i>Importing Libraries</i></h2>
<hr>


```python
import numpy as np
from time import time
from typing import Any, List, Tuple, TypedDict
from nptyping import NDArray, Float64
np.seterr( over = 'raise' )
```




    {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}



<h2 style="font-size:32px; color:#F4EE00; font-family:cambria"><i>Defining Data Structures</i></h2>
<hr>

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Input Matrix</span></i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{blue}{X} = \begin{bmatrix}
    \color{blue}{x_{1,1}} & \color{blue}{x_{1,2}} & \dots & \color{blue}{x_{1,n}} \\
    \color{blue}{x_{2,1}} & \color{blue}{x_{2,2}} & \dots & \color{blue}{x_{2,n}} \\
    \vdots & \vdots & \ddots & \vdots \\
    \color{blue}{x_{m,1}} & \color{blue}{x_{m,2}} & \dots & \color{blue}{x_{m,n}} \\
\end{bmatrix}
$</span>


```python
Examples = Any
Features = Any
Input_Matrix = NDArray[ ( Examples, Features ), Float64 ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Target Matrix</span></i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{yellow}{Y} = \begin{bmatrix}
    \color{yellow}{y_{1,1}} & \color{yellow}{y_{1,2}} & \dots & \color{yellow}{y_{1,k}} \\
    \color{yellow}{y_{2,1}} & \color{yellow}{y_{2,2}} & \dots & \color{yellow}{y_{2,k}} \\
    \vdots & \vdots & \ddots & \vdots \\
    \color{yellow}{y_{m,1}} & \color{yellow}{y_{m,2}} & \dots & \color{yellow}{y_{m,k}}
\end{bmatrix}
$</span>


```python
Targets = Any
Target_Matrix = NDArray[ ( Examples, Targets ), Float64 ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Output Matrix</span></i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{yellow}{\hat{Y}} = \begin{bmatrix}
    \color{yellow}{\hat{y}_{1,1}} & \color{yellow}{\hat{y}_{1,2}} & \dots & \color{yellow}{\hat{y}_{1,k}} \\
    \color{yellow}{\hat{y}_{2,1}} & \color{yellow}{\hat{y}_{2,2}} & \dots & \color{yellow}{\hat{y}_{2,k}} \\
    \vdots & \vdots & \ddots & \vdots \\
    \color{yellow}{\hat{y}_{m,1}} & \color{yellow}{\hat{y}_{m,2}} & \dots & \color{yellow}{\hat{y}_{m,k}}
\end{bmatrix}
$


```python
Output_Matrix = NDArray[ ( Examples, Targets ), Float64 ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Perceptron</span></i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{green}{P}^{(ℓ)}_{\color{green}{k}} = \begin{bmatrix}
    \color{green}{\omega}^{(ℓ)}_{\color{green}{1,k}} \\
    \color{green}{\omega}^{(ℓ)}_{\color{green}{2,k}} \\
    \vdots \\ 
    \color{green}{\omega}^{(ℓ)}_{\color{green}{j,k}}
\end{bmatrix} \\
\color{green}{b}^{(ℓ)}_{\color{green}{k}} = \color{green}{\omega}^{(ℓ)}_{\color{green}{0,k}}
$


```python
Inputs = Any
Bias, Weight = Float64, Float64
Perceptron = NDArray[ ( Inputs, 1 ), Weight ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Layer</span></i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{green}{\Omega}^{(ℓ)} = \begin{bmatrix}
    \color{green}{P}^{(ℓ)}_{\color{green}{1}} & 
    \color{green}{P}^{(ℓ)}_{\color{green}{2}} & 
    \dots & 
    \color{green}{P}^{(ℓ)}_{\color{green}{k}}
\end{bmatrix} \\
\color{green}{\beta}^{(ℓ)} = \begin{bmatrix}
    \color{green}{b}^{(ℓ)}_{\color{green}{1}} & 
    \color{green}{b}^{(ℓ)}_{\color{green}{2}} & 
    \dots & 
    \color{green}{b}^{(ℓ)}_{\color{green}{k}}
\end{bmatrix}
$</span>


```python
Number_of_Perceptrons = Any
Weights = NDArray[ ( Inputs, Number_of_Perceptrons ), Perceptron ]
Biases = NDArray[ Number_of_Perceptrons, Bias ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Network</span></i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{green}{N} = \begin{bmatrix}
    \begin{bmatrix}
       \color{green}{\Omega}^{(1)} &
       \color{green}{\Omega}^{(2)} & 
       \dots & 
       \color{green}{\Omega}^{(ℓ)}
   \end{bmatrix} & 
   \begin{bmatrix}
       \color{green}{\beta}^{(1)} &
       \color{green}{\beta}^{(2)} & 
       \dots & 
       \color{green}{\beta}^{(ℓ)}
   \end{bmatrix}
\end{bmatrix}
$</span>


```python
Network_Weights = List[ Weights ]
Network_Biases = List[ Biases ]
class Network( TypedDict ) :
    weights : Network_Weights
    biases  : Network_Biases
```

<h1 style="font-size:32px; color:#F4EE00; font-family:cambria"><i>Creating the Neural Network</i></h1>
<hr>

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Initialization</span></i>
</span>


```python
class FeedForwardNeuralNetwork :
    
    def __init__( self, perceptrons_per_hidden_layer : List[ int ] = [] ) -> None :
        self.score = 0.0
        self.perlayer = perceptrons_per_hidden_layer
        self.network : Network = { 'weights' : [], 'biases' : [] }
        return
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Initialize Random Weights and Biases</span></i>
</span>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def initialize( self, X : Input_Matrix, Y : Target_Matrix ) -> None :
        inputs = X.shape[ 1 ]
        self.network = { 'weights' : [], 'biases' : [] }
        for perceptrons in self.perlayer :
            self.network[ 'weights' ].append( np.random.rand( inputs, perceptrons ) - 0.5 )
            self.network[ 'biases' ].append( np.random.rand( perceptrons ) - 0.5 )
            inputs = perceptrons
        self.network[ 'weights' ].append( np.random.rand( inputs, Y.shape[ 1 ] ) - 0.5 )
        self.network[ 'biases' ].append( np.random.rand( Y.shape[ 1 ] ) - 0.5 )
        return
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Activation Function</span></i>
</span>
<br>
<br>
<table align="left">
    <tr>
        <td style="font-size:24px">$\color{blue}{Sigmoid}$</td>
        <td style="font-size:24px; text-align:left">$
        \color{pink}{f}(x) = \frac{1}{1+\color{blue}{e}^{-x}}
        $</td>
    </tr>
    <tr>
        <td style="font-size:24px">$\color{blue}{Derivative}$</td>
        <td style="font-size:24px">$
        \color{pink}{f}(x) = \frac{\color{blue}{e}^{-x}}{(1+\color{blue}{e}^{-x})^{2}}
        $</td>
    </tr>
</table>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def activation( self, x : NDArray[ Float64 ] ) -> NDArray[ Float64 ] :
        return 1.0 / ( 1.0 + np.exp( -x ) )
    
    def derivative( self, x : NDArray[ Float64 ] ) -> NDArray[ Float64 ] :
        return np.exp( -x ) / np.square( 1.0 + np.exp( -x ) )
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Cost and Gradient</span></i>
</span>
<br>
<br>
<table align="left">
    <tr>
        <td style="font-size:24px">$\color{blue}{Cost\ Function}$</td>
        <td style="font-size:24px">$
        \color{pink}{C}(\color{yellow}{Y};\color{yellow}{\hat{Y}}) = 
        \sum_{h=1}^{m}{\sum_{i=1}^{k} (\color{yellow}{\hat{y}}_{h,i} - \color{yellow}{y}_{h,i}})^{2} =
        \sum_{h=1}^{m}{\sum_{i=1}^{k} (\color{pink}{a}^{(ℓ)}_{h,i} - \color{yellow}{y}_{h,i}})^{2} 
        $</td>
    </tr>
    <tr>
        <td style="font-size:24px">$\color{blue}{Gradient}$</td>
        <td style="font-size:24px; text-align:left">$
        \nabla^{(ℓ)}_{\color{pink}{a}}\color{pink}{C} = \begin{bmatrix}
                \frac{\partial \color{pink}{C}}{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{1,1}}} &
                \frac{\partial \color{pink}{C}}{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{1,2}}} &
                \dots & 
                \frac{\partial \color{pink}{C}}{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{1,k}}} 
                \\
                \frac{\partial \color{pink}{C}}{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{2,1}}} &
                \frac{\partial \color{pink}{C}}{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{2,2}}} &
                \dots & 
                \frac{\partial \color{pink}{C}}{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{2,k}}}
                \\
                \vdots & \vdots & \ddots & \vdots \\
                \\
                \frac{\partial \color{pink}{C}}{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{m,1}}} &
                \frac{\partial \color{pink}{C}}{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{m,2}}} &
                \dots & 
                \frac{\partial \color{pink}{C}}{\partial \color{pink}{a}^{(ℓ)}_{\color{pink}{m,k}}}
            \end{bmatrix} = 2(\color{yellow}{\hat{Y}}-\color{yellow}{Y}) 
        $</td>
    </tr>
</table>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def cost( self, A : Output_Matrix, Y : Target_Matrix ) -> Float64 :
        return np.square( A - Y ).sum()
    
    def grad( self, A : Output_Matrix, Y : Target_Matrix ) -> NDArray[ Float64 ] :
        return 2*( A - Y )
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Forward Propagation</span></i>
</span>
<br>
<br>
<table align="left">
    <tr>
        <td style="font-size:24px">$\color{blue}{Output\ and\ Hidden\ Layer}$</td>
        <td style="font-size:24px">$
            \color{blue}{L}^{(ℓ)}=
            \color{red}{f}⊙(\color{blue}{L}^{(ℓ-1)}\color{green}{\Omega}^{(ℓ)} + \color{green}{\beta}^{(ℓ)})
            $
        </td>
    </tr>
    <tr>
        <td style="font-size:24px">$\color{blue}{Input Layer}$</td>
        <td style="font-size:24px; text-align:left">$\color{blue}{L}^{(0)} = \color{blue}{X}$</td>
    </tr>
</table>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def forwardpropagation( self, L : Input_Matrix ) -> Output_Matrix :
        for w, b in zip( self.network[ 'weights' ], self.network[ 'biases' ] ) :
            L = self.activation( np.matmul( L, w ) + b ) 
        return L
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Backpropagation</span></i>
</span>
<br>
<br>
<table align="left">
    <tr>
        <td style="font-size:24px">$\color{blue}{Containers}$</td>
        <td style="font-size:24px; text-align:left">
            $
            \color{orange}{Z}[ℓ] = \color{blue}{L}^{(ℓ-1)}\color{green}{\Omega}^{(ℓ)}+\color{green}{\beta}^{(ℓ)} \\
            \color{pink}{A}[ℓ] = \color{blue}{L}^{(ℓ)}
            $
        </td>
    </tr>
</table>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def __forwardpropagation( self, X : Input_Matrix, A : List, Z : List ) -> Output_Matrix :
        A.append( X ) # len( A ) = len( Z ) + 1
        for w, b in zip( self.network[ 'weights' ], self.network[ 'biases' ] ) :
            # weighted input to layer
            Z.append( np.matmul( A[ -1 ], w ) + b )
            # output of layer
            A.append( self.activation( Z[ -1 ] ) )
        return A[ -1 ]
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i><span style="font-size:32px">Backpropagation</span></i>
</span>
<br>
<br>
<table align="left">
    <tr>
        <td style="font-size:24px">$\color{blue}{Initialization}$</td>
        <td style="font-size:24px; text-align:left">
            $
            \nabla^{(ℓ)}_{\color{orange}{z}}\color{pink}{C}=
            (\color{pink}{f}'⊙\color{orange}{Z}[ℓ])⊙\nabla^{(ℓ)}_{\color{pink}{a}}\color{pink}{C}
            $
        </td>
    </tr>
    <tr>
        <td style="font-size:24px">$\color{blue}{Output\ and\ Hidden\ Layer}$</td>
        <td style="font-size:24px">
            $
            \nabla^{(ℓ)}_{\color{green}{\Omega}}\color{pink}{C}=
            \color{pink}{A}[ℓ-1]^{T}\nabla^{(ℓ)}_{\color{orange}{z}}\color{pink}{C} \\
            \nabla^{(ℓ)}_{\color{green}{\beta}}\color{pink}{C}=
            \pmb{1}_{1\times m}\nabla^{(ℓ)}_{\color{orange}{z}}\color{pink}{C} \\
            \nabla^{(ℓ-1)}_{\color{orange}{z}}\color{pink}{C}=
            (\color{pink}{f}'⊙\color{orange}{Z}[ℓ-1])⊙\nabla^{(ℓ)}_{\color{orange}{z}}\color{pink}{C}(\color{green}{\Omega}^{(ℓ)})^{T}
            $
        </td>
    </tr>
</table>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def backpropagate( self, grad_z : NDArray[ Float64 ], A : List, Z : List, layer_index : int ) -> Tuple :
        # gradient with respect to the weights of the layer
        grad_w = np.matmul( A[ layer_index ].T, grad_z ) # len( A ) = len( Z ) + 1
        # gradient with respect to the biases of the layer
        grad_b = grad_z.sum( axis = 0 )
        # gradient with respect to the weighted input of the layer
        if layer_index > 0 : # there is no weighted input for layer 0
            grad_z = self.derivative( Z[ layer_index - 1 ] ) *\
                     np.matmul( grad_z, self.network[ 'weights' ][ layer_index ].T )
        return grad_z, grad_w, grad_b
```

<span style="color:#53C8FE; font-family:aparajita; font-size:18px">
    <i>Gradient Descent</i>
</span>
<br>
<br>
<span style="font-size:24px">$
\color{green}{\Omega}^{(ℓ)} \rightarrow \color{green}{\Omega}^{(ℓ)} - \frac{r}{m}\nabla^{(ℓ)}_{\color{green}{\Omega}}\color{pink}{C} \\
\color{green}{\beta}^{(ℓ)} \rightarrow \color{green}{\beta}^{(ℓ)} - \frac{r}{m}\nabla^{(ℓ)}_{\color{green}{\beta}}\color{pink}{C}
$</span>


```python
class FeedForwardNeuralNetwork( FeedForwardNeuralNetwork  ) :
    
    def train( self, X : Input_Matrix, Y : Target_Matrix, 
               learning_rate = 1.0, convergence = 0.01, 
               batch_size = 10, max_epoch = 500, max_time = 60 ) -> None :
        ''' Stochastic Gradient Descent '''
        epoch = 1
        start = time()
        totgrad = np.inf
        self.initialize( X, Y )
        output_layer_index = len( self.network[ 'weights' ] ) - 1
        while np.sqrt( totgrad ) > convergence :
            totgrad = 0
            shuffle = np.random.permutation( len( X ) )
            X, Y = X[ shuffle ], Y[ shuffle ]
            for batch_x, batch_y in zip( 
                np.array_split( X, len( X ) // batch_size ), 
                np.array_split( Y, len( Y ) // batch_size ) 
                ) :
                A, Z = [], []
                output = self.__forwardpropagation( batch_x, A, Z )
                # gradient with respect to the output layer
                grad_a = self.grad( output, batch_y )
                totgrad += np.linalg.norm( grad_a )**2
                # gradient with respect to the weighted input of the output layer
                grad_z = self.derivative( Z[ -1 ] ) * grad_a
                for layer_index in range( output_layer_index, -1, -1 ) :
                    grad_z, grad_w, grad_b = self.backpropagate( grad_z, A, Z, layer_index )
                    # updating the weights and biases of layer
                    self.network[ 'weights' ][ layer_index ] -= learning_rate * grad_w / len( batch_x )
                    self.network[ 'biases' ][ layer_index ] -= learning_rate * grad_b / len( batch_x )
            epoch += 1
            if time() - start > max_time :
                print( 'Maximum runtime encountered.' )
                break
            if epoch > max_epoch :
                print( 'Maximum epoch encountered.' )
                break
        self.score = self.cost( self.forwardpropagation( X ), Y )
        return
```
