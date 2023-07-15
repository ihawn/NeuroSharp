# NeuroSharp

NeuroSharp is a C# neural network framework designed for building and training neural networks. It provides a set of flexible and extensible classes for constructing various types of neural networks, including activation layers, convolutional layers, fully connected (dense) layers, LSTM layers, max pooling layers, parameterized layers, recurrent layers, and softmax activation layers.

# Getting Started

To get started with NeuroSharp, you can follow the example code provided below:
```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp;
using NeuroSharp.Data;
using NeuroSharp.Enumerations;
using NeuroSharp.Training;

namespace Trainer
{
    public class Trainer
    {
        static void Main(string[] args)
        {
            // Example usage: LetterIdentificationTraining(20);
            SentimentAnalysisTraining(15, trainingSize: 50000, testSize: 5000, maxWordCount: 1500, maxReviewLength: 20);
        }

        static void LetterIdentificationTraining(int epochs)
        {
            // Prepare the training data
            string path = @"path";
            string possibleChars = "abcdefghijklmnopqrst";

            List<(Vector<double>, Vector<double>)> data = new List<(Vector<double>, Vector<double>)>();

            foreach (string file in Directory.EnumerateFiles(path, "*.txt"))
            {
                // Load the input data and label for each file
                Vector<double> x = Vector<double>.Build.DenseOfEnumerable(
                    File.ReadAllText(file).Split(",").Select(x => double.Parse(x))
                );

                string character = file.Replace(path + @"\", "")[0].ToString();
                Vector<double> y = Vector<double>.Build.Dense(possibleChars.Length);
                y[possibleChars.IndexOf(character)] = 1;

                data.Add((x, y));
            }

            // Shuffle the data randomly
            Random rand = new Random();
            data = data.OrderBy(x => rand.Next()).ToList();

            double trainSplit = 1;

            List<Vector<double>> xTrain =
                data.Take((int)Math.Round(trainSplit * data.Count)).Select(x => x.Item1).ToList();
            List<Vector<double>> yTrain =
                data.Take((int)Math.Round(trainSplit * data.Count)).Select(y => y.Item2).ToList();

            List<Vector<double>> xTest =
                data.Skip((int)Math.Round(trainSplit * data.Count)).Select(x => x.Item1).ToList();
            List<Vector<double>> yTest =
                data.Skip((int)Math.Round(trainSplit * data.Count)).Select(y => y.Item2).ToList();

            // Build the network architecture
            Network network = new Network(xTrain[0].Count);
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 128, stride: 1, channels: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 64, stride: 1, channels: 128));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 16, stride: 2, channels: 64));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new FullyConnectedLayer(512));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(128));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(64));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(yTrain[0].Count));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            // Train the network
            network.Train(xTrain, yTrain, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.002f);

            // Save the trained model
            string modelJson = network.SerializeToJSON();
            File.WriteAllText(@"path\characters_model.json", modelJson);

            if (trainSplit < 1)
            {
                // Evaluate the accuracy on the test set
                int i = 0;
                int wrongCount = 0;
                foreach (var test in xTest)
                {
                    var output = network.Predict(test);
                    int prediction = output.ToList().IndexOf(output.Max());
                    int actual = yTest[i].ToList().IndexOf(yTest[i].Max());
                    Console.WriteLine("Prediction: " + prediction);
                    Console.WriteLine("Actual: " + actual + "\n");

                    if (prediction != actual)
                        wrongCount++;

                    i++;
                }

                double acc = (1d - ((double)wrongCount) / i);
                Console.WriteLine("Accuracy: " + acc);
            }
        }

        static void SentimentAnalysisTraining(int epochs, int trainingSize, int testSize, int maxWordCount, int maxReviewLength)
        {
            // TODO: Add code for sentiment analysis training
        }
    }
}
```
# Layers

NeuroSharp supports various types of layers that can be added to the neural network. Here are the available layer types:

- Activation Layer: Applies an activation function to the input.
- Convolutional Layer: Performs convolution operation on the input data.
- Fully Connected Layer: Connects every neuron from the previous layer to the current layer.
- LSTM Layer: Long Short-Term Memory layer for sequence modeling.
- Max Pooling Layer: Performs max pooling operation on the input data.
- Parameterized Layer: A base class for layers with learnable parameters.
- Recurrent Layer: Recurrent layer for sequence modeling.
- Softmax Activation Layer: Applies softmax activation to the input.

You can add layers to the network using the Add method of the Network class.

# Training

NeuroSharp provides three training configurations: SGD (Stochastic Gradient Descent), Mini-batch, and Batch training. You can choose the training configuration by passing the appropriate TrainingConfiguration enum value to the Train method of the Network class.


```csharp
// Example: Training using mini-batch with Adam optimizer
network.Train(xTrain, yTrain, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.002f);
```

# Saving and Loading Models

You can save a trained model using the SerializeToJSON method of the Network class and save it to a JSON file.

```csharp
string modelJson = network.SerializeToJSON();
File.WriteAllText("model.json", modelJson);
```

To load a saved model, you can use the DeserializeNetworkJSON method of the Network class.

```csharp
string modelJson = File.ReadAllText("model.json");
Network network = Network.DeserializeNetworkJSON(modelJson);
```

# Contributions and Feedback

NeuroSharp is an open-source project, and contributions are welcome. If you have any feedback, suggestions, or bug reports, please create an issue on the project's GitHub repository.
