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

        [Test]
        public void CategoricalCrossentropy_ReturnsCorrectInputGradient_WhenChainedTogetherWithSoftmaxTanhAndConvolutionalLayer_KernelSameSizeAsImageSoNoStride()
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
        public void CategoricalCrossentropy_ReturnsCorrectInputGradient_WhenChainedTogetherWithSoftmaxTanhAndConvolutionalLayer_WithStride1()
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
    }
}
