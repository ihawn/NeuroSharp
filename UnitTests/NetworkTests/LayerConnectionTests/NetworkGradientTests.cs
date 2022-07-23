using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;


namespace UnitTests.NetworkTests.LayerConnectionTests
{
    public class NetworkGradientTests
    {
        [SetUp]
        public void Setup()
        {
        }
        #region Activation And Loss Gradient Tests
        [Test]
        public void CategoricalCrossentropy_ReturnsCorrectInputGradient_WhenChainedTogetherWithSoftmaxActivation()
        {
            for (int i = 1; i < 100; i++)
            {
                Vector<double> truthY = Vector<double>.Build.Random(i);
                Vector<double> testX = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                double networkLoss(Vector<double> x)
                {
                    x = network.Predict(x);
                    return network.Loss(truthY, x);
                }

                Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
                Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    testGradient = network.Layers[k].BackPropagation(testGradient, OptimizerType.Adam, 1, 0.0001);
                }

                Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
            }
        }

        [Test]
        public void CategoricalCrossentropy_ReturnsCorrectInputGradient_WhenChainedTogetherWithSoftmaxAndTanhActivation()
        {
            for (int i = 1; i < 100; i++)
            {
                Vector<double> truthY = Vector<double>.Build.Random(i);
                Vector<double> testX = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                double networkLoss(Vector<double> x)
                {
                    x = network.Predict(x);
                    return network.Loss(truthY, x);
                }

                Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
                Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    testGradient = network.Layers[k].BackPropagation(testGradient, OptimizerType.Adam, 1, 0.0001);
                }

                Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
            }
        }

        [Test]
        public void CategoricalCrossentropy_ReturnsCorrectInputGradient_WhenChainedTogetherWithSoftmaxAndReluActivation()
        {
            for (int i = 1; i < 100; i++)
            {
                Vector<double> truthY = Vector<double>.Build.Random(i);
                Vector<double> testX = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                double networkLoss(Vector<double> x)
                {
                    x = network.Predict(x);
                    return network.Loss(truthY, x);
                }

                Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
                Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    testGradient = network.Layers[k].BackPropagation(testGradient, OptimizerType.Adam, 1, 0.0001);
                }

                Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
            }
        }
        #endregion

        #region Convolutional Gradient Tests
        //Categorical Crossentropy Loss
        [Test]
        public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_KernelSameSizeAsImageSoNoStride()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 25; i++)
                squares.Add((int)Math.Pow(i, 2));
            foreach(int i in squares)
            {
                Vector<double> truthY = Vector<double>.Build.Random(1);
                Vector<double> testX = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(i), filters: 1, stride: 1));
                network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                double networkLoss(Vector<double> x)
                {
                    x = network.Predict(x);
                    return network.Loss(truthY, x);
                }

                Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
                Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    testGradient = network.Layers[k].BackPropagation(testGradient, OptimizerType.Adam, 1, 0.0001);
                }

                Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
            }
        }

        [Test]
        public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_WithStride1()
        {            
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));

            foreach(int i in squares)
            {
                foreach(int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    int outdim = (int)Math.Floor(Math.Sqrt(i) - Math.Sqrt(j)) + 1;
                    Vector<double> truthY = Vector<double>.Build.Random(outdim*outdim);
                    Vector<double> testX = Vector<double>.Build.Random(i);

                    Network network = new Network();
                    network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: 1, stride: 1));
                    network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                    network.Add(new SoftmaxActivationLayer());
                    network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                    double networkLoss(Vector<double> x)
                    {
                        x = network.Predict(x);
                        return network.Loss(truthY, x);
                    }

                    Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
                    Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
                    for (int k = network.Layers.Count - 1; k >= 0; k--)
                    {
                        testGradient = network.Layers[k].BackPropagation(testGradient, OptimizerType.Adam, 1, 0.0001);
                    }

                    Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
                }
            }
        }

        [Test]
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_KernelSameSizeAsImageSoNoStride()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));
            foreach(int i in squares)
            {
                Vector<double> truthY = Vector<double>.Build.Random(1);
                Vector<double> testX = Vector<double>.Build.Random(i);
                Vector<double> testWeight = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(i), filters: 1, stride: 1));
                network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                double networkLossWithWeightAsVariable(Vector<double> x)
                {
                    ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                    conv.Weights[0] = Utils.Unflatten(x);
                    Vector<double> output = network.Predict(testX);
                    return network.Loss(truthY, output);
                }

                Vector<double> finiteDiffWeightGradient = Utils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeight);
                Vector<double> explicitWeightGradient = Vector<double>.Build.Dense(i);

                Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    outputGradient = network.Layers[k].BackPropagation(outputGradient, OptimizerType.Adam, 1, 0.0001);
                    if (k == 0) // retrieve weight gradient from convolutional layer
                        explicitWeightGradient = Utils.Flatten(((ConvolutionalLayer)network.Layers[0]).WeightGradient[0]);
                }

                Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.00001);
            }
        }

        [Test]
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_WithStride1()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    int outdim = (int)Math.Floor(Math.Sqrt(i) - Math.Sqrt(j)) + 1;
                    Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim);
                    Vector<double> testX = Vector<double>.Build.Random(i);
                    Vector<double> testWeight = Vector<double>.Build.Random(j);

                    Network network = new Network();
                    network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: 1, stride: 1));
                    network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                    network.Add(new SoftmaxActivationLayer());
                    network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                    double networkLossWithWeightAsVariable(Vector<double> x)
                    {
                        ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                        conv.Weights[0] = Utils.Unflatten(x);
                        Vector<double> output = network.Predict(testX);
                        return network.Loss(truthY, output);
                    }

                    Vector<double> finiteDiffWeightGradient = Utils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeight);
                    Vector<double> explicitWeightGradient = Vector<double>.Build.Dense(j);

                    Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                    for (int k = network.Layers.Count - 1; k >= 0; k--)
                    {
                        outputGradient = network.Layers[k].BackPropagation(outputGradient, OptimizerType.Adam, 1, 0.0001);
                        if (k == 0) // retrieve weight gradient from convolutional layer
                            explicitWeightGradient = Utils.Flatten(((ConvolutionalLayer)network.Layers[0]).WeightGradient[0]);
                    }

                    Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.0001);
                }
            }
        }

        //Mean Squared Error Loss
        /*[Test]
        public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithMSESoftmaxTanhAndConvolutionalLayer_KernelSameSizeAsImageSoNoStride()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 25; i++)
                squares.Add((int)Math.Pow(i, 2));
            foreach (int i in squares)
            {

            }
            Vector<double> truthY = Vector<double>.Build.Random(1);
            Vector<double> testX = Vector<double>.Build.Random(4);

            Network network = new Network();
            network.Add(new ConvolutionalLayer(4, kernel: (int)Math.Sqrt(4), filters: 1, stride: 1));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

            double networkLoss(Vector<double> x)
            {
                x = network.Predict(x);
                return network.Loss(truthY, x);
            }

            Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
            Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
            for (int k = network.Layers.Count - 1; k >= 0; k--)
            {
                testGradient = network.Layers[k].BackPropagation(testGradient, OptimizerType.Adam, 1, 0.0001);
            }

            Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
        }*/

        /*[Test]
        public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithMSESoftmaxTanhAndConvolutionalLayer_WithStride1()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    int outdim = (int)Math.Floor(Math.Sqrt(i) - Math.Sqrt(j)) + 1;
                    Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim);
                    Vector<double> testX = Vector<double>.Build.Random(i);

                    Network network = new Network();
                    network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: 1, stride: 1));
                    network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                    network.Add(new SoftmaxActivationLayer());
                    network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                    double networkLoss(Vector<double> x)
                    {
                        x = network.Predict(x);
                        return network.Loss(truthY, x);
                    }

                    Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
                    Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
                    for (int k = network.Layers.Count - 1; k >= 0; k--)
                    {
                        testGradient = network.Layers[k].BackPropagation(testGradient, OptimizerType.Adam, 1, 0.0001);
                    }

                    Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
                }
            }
        }

        [Test]
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithMSESoftmaxTanhAndConvolutionalLayer_KernelSameSizeAsImageSoNoStride()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));
            foreach (int i in squares)
            {
                Vector<double> truthY = Vector<double>.Build.Random(1);
                Vector<double> testX = Vector<double>.Build.Random(i);
                Vector<double> testWeight = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(i), filters: 1, stride: 1));
                network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                double networkLossWithWeightAsVariable(Vector<double> x)
                {
                    ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                    conv.Weights[0] = Utils.Unflatten(x);
                    Vector<double> output = network.Predict(testX);
                    return network.Loss(truthY, output);
                }

                Vector<double> finiteDiffWeightGradient = Utils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeight);
                Vector<double> explicitWeightGradient = Vector<double>.Build.Dense(i);

                Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    outputGradient = network.Layers[k].BackPropagation(outputGradient, OptimizerType.Adam, 1, 0.0001);
                    if (k == 0) // retrieve weight gradient from convolutional layer
                        explicitWeightGradient = Utils.Flatten(((ConvolutionalLayer)network.Layers[0]).WeightGradient[0]);
                }

                Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.00001);
            }
        }

        [Test]
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithMSESoftmaxTanhAndConvolutionalLayer_WithStride1()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    int outdim = (int)Math.Floor(Math.Sqrt(i) - Math.Sqrt(j)) + 1;
                    Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim);
                    Vector<double> testX = Vector<double>.Build.Random(i);
                    Vector<double> testWeight = Vector<double>.Build.Random(j);

                    Network network = new Network();
                    network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: 1, stride: 1));
                    network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                    network.Add(new SoftmaxActivationLayer());
                    network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                    double networkLossWithWeightAsVariable(Vector<double> x)
                    {
                        ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                        conv.Weights[0] = Utils.Unflatten(x);
                        Vector<double> output = network.Predict(testX);
                        return network.Loss(truthY, output);
                    }

                    Vector<double> finiteDiffWeightGradient = Utils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeight);
                    Vector<double> explicitWeightGradient = Vector<double>.Build.Dense(j);

                    Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                    for (int k = network.Layers.Count - 1; k >= 0; k--)
                    {
                        outputGradient = network.Layers[k].BackPropagation(outputGradient, OptimizerType.Adam, 1, 0.0001);
                        if (k == 0) // retrieve weight gradient from convolutional layer
                            explicitWeightGradient = Utils.Flatten(((ConvolutionalLayer)network.Layers[0]).WeightGradient[0]);
                    }

                    Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.0001);
                }
            }
        }*/
        #endregion
    }
}
