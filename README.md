# Redes Neurais

Implementações de redes neurais em C.
Para todos os casos é usado MNIST.

## Redes implementadas
- MLP (Multi-Layer Perceptron)
- CNN (Convolutional Neural Network)
- RNN (Recurrent Neural Network)
- GRNN(GRU-RNN) (Gated Recurrent Unit Neural Network)

## Em desenvolvimento
- Batch-norm para todas as redes (exceto RNN)
- Camadas Residuais para CNN
  
## Compilação e Execução

```bash
# Clone o repositório
git clone https://github.com/MichelHilarioDeLucena/redes_neurais.git
cd redes_neurais/NEURAL_NETS

# Compile
cmake -B build
cmake --build build

# Execute

./build/mlp
./build/cnn
./build/rnn
./build/grnn
```
