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
        
        [Test]
        public void LSTM_BackPropagation_ReturnsCorrectHGradient()
        {
            int inputUnits = 2;
            int outputUnits = 2;
            int hiddenUnits = 2;
            int sequenceLength = 2;
            
            Vector<double> truthY = Vector<double>.Build.Random(outputUnits * (sequenceLength - 1));
            Vector<double> testX = Vector<double>.Build.Random(outputUnits * sequenceLength);
            Vector<double> testWeight = Vector<double>.Build.Random(hiddenUnits * outputUnits);

            Network network = new Network(outputUnits * sequenceLength);
            network.Add(new LongShortTermMemoryLayer(inputUnits, outputUnits, hiddenUnits, sequenceLength));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            double networkLossHWeightInput(Vector<double> x)
            {
                LongShortTermMemoryLayer lstm = (LongShortTermMemoryLayer)network.Layers[0];
                lstm.Weights[(int)LSTMWeight.H] = MathUtils.Unflatten(x, hiddenUnits, outputUnits);
                x = network.Predict(testX);
                return network.Loss(truthY, x);
            }

            Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLossHWeightInput, testWeight);
            Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
            Vector<double> explicitWeightGradient = null;
            for (int k = network.Layers.Count - 1; k >= 0; k--)
            {
                testGradient = network.Layers[k].BackPropagation(testGradient);
                if (k == 0)
                {
                    LongShortTermMemoryLayer lstm = (LongShortTermMemoryLayer)network.Layers[0];
                    explicitWeightGradient = MathUtils.Flatten(lstm.WeightGradients[(int)LSTMWeight.H]);
                }
            }

            Assert.IsTrue((finiteDiffGradient - explicitWeightGradient).L2Norm() < 0.00001);
        }
    }
}

//todo train batching by stacking vector batches into a matrix