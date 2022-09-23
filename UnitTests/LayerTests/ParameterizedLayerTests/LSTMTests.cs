using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Datatypes;
using NeuroSharp.Training;

namespace UnitTests.LayerTests.ParameterizedLayerTests
{
    internal class LSTMTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void LSTM_ForwardPropagation_NoRuntimeErrors()
        {
            Vector<double> x = Vector<double>.Build.Random(27 * 12);

            LongShortTermMemoryLayer layer = new LongShortTermMemoryLayer(100, 27, 256, 12);
            layer.InitializeParameters();

            Vector<double> output = layer.ForwardPropagation(x);
        }

        [Test]
        public void LSTM_BackPropagation_ReturnsCorrectInputGradient()
        {
            Vector<double> truthY = Vector<double>.Build.Random(27 * 11);
            Vector<double> testX = Vector<double>.Build.Random(27 * 12);

            Network network = new Network(27 * 12);
            network.Add(new LongShortTermMemoryLayer(100, 27, 256, 12));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            double networkLoss(Vector<double> x)
            {
                x = network.Predict(x);
                return network.Loss(truthY, x);
            }

            Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLoss, testX);
            Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
            for (int k = network.Layers.Count - 1; k >= 0; k--)
            {
                testGradient = network.Layers[k].BackPropagation(testGradient);
            }

            Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
        }
    }
}
