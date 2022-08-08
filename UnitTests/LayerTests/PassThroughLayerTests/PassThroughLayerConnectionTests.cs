using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Training;

namespace UnitTests.NetworkTests.LayerConnectionTests
{
    public class PassThroughLayerConnectionTests
    {
        [SetUp]
        public void Setup()
        {
        }

        #region Categorical Crossentropy Connection Tests
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
                    testGradient = network.Layers[k].BackPropagation(testGradient);
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
                    testGradient = network.Layers[k].BackPropagation(testGradient);
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
                    testGradient = network.Layers[k].BackPropagation(testGradient);
                }

                Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
            }
        }
        #endregion

        #region Mean SquaredError Connection Tests
        [Test]
        public void MSE_ReturnsCorrectInputGradient_WhenChainedTogetherWithSoftmaxActivation()
        {
            for (int i = 1; i < 100; i++)
            {
                Vector<double> truthY = Vector<double>.Build.Random(i);
                Vector<double> testX = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

                double networkLoss(Vector<double> x)
                {
                    x = network.Predict(x);
                    return network.Loss(truthY, x);
                }

                Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
                Vector<double> testGradient = LossFunctions.MeanSquaredErrorPrime(truthY, network.Predict(testX));
                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    testGradient = network.Layers[k].BackPropagation(testGradient);
                }

                Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
            }
        }

        [Test]
        public void MSE_ReturnsCorrectInputGradient_WhenChainedTogetherWithSoftmaxAndTanhActivation()
        {
            for (int i = 1; i < 100; i++)
            {
                Vector<double> truthY = Vector<double>.Build.Random(i);
                Vector<double> testX = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

                double networkLoss(Vector<double> x)
                {
                    x = network.Predict(x);
                    return network.Loss(truthY, x);
                }

                Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
                Vector<double> testGradient = LossFunctions.MeanSquaredErrorPrime(truthY, network.Predict(testX));
                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    testGradient = network.Layers[k].BackPropagation(testGradient);
                }

                Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
            }
        }

        [Test]
        public void MSE_ReturnsCorrectInputGradient_WhenChainedTogetherWithSoftmaxAndReluActivation()
        {
            for (int i = 1; i < 100; i++)
            {
                Vector<double> truthY = Vector<double>.Build.Random(i);
                Vector<double> testX = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

                double networkLoss(Vector<double> x)
                {
                    x = network.Predict(x);
                    return network.Loss(truthY, x);
                }

                Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
                Vector<double> testGradient = LossFunctions.MeanSquaredErrorPrime(truthY, network.Predict(testX));
                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    testGradient = network.Layers[k].BackPropagation(testGradient);
                }

                Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
            }
        }
        #endregion
    }
}
