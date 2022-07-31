using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;

namespace UnitTests.LayerTests.ParameterizedLayerTests
{
    internal class DenseLayerTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void DenseOnlyLayer_PropagatesCorrectGradientBackwards()
        {
            for(int i = 1; i < 50; i++)
            {
                for(int j = 1; j < 50; j++)
                {
                    Vector<double> truthY = Vector<double>.Build.Random(j);
                    Vector<double> testX = Vector<double>.Build.Random(i);

                    Network network = new Network();
                    network.Add(new FullyConnectedLayer(i, j));
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

                    Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.03);
                }
            }
        }
    }
}
