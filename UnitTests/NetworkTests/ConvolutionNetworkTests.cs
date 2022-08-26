using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Data;
using NeuroSharp.Training;

namespace UnitTests
{
    public class ConvolutionalNetworkTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void ChainedConvolutionalLayers_PropogateCorrectInputGradients()
        {
            Vector<double> trainX = Vector<double>.Build.Random(56 * 56 * 2);
            Vector<double> truthY = Vector<double>.Build.Dense(4);

            //note that the number of filters on a convolutional layer turns into the number of channels on the next one
            Network network = new Network(56 * 56 * 2);
            network.Add(new MultiChannelConvolutionalLayer(56 * 56 * 2, kernel: 8, filters: 6, stride: 2, channels: 2));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new MultiChannelConvolutionalLayer(25 * 25 * 6, kernel: 4, filters: 5, stride: 3, channels: 6));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MultiChannelConvolutionalLayer(8 * 8 * 5, kernel: 2, filters: 7, stride: 2, channels: 5));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(4 * 4 * 7, prevFilterCount: 7, poolSize: 2));
            network.Add(new FullyConnectedLayer(10));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(4));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            double networkLoss(Vector<double> x)
            {
                x = network.Predict(x);
                return network.Loss(truthY, x);
            }

            Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLoss, trainX);
            Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(trainX));
            for (int k = network.Layers.Count - 1; k >= 0; k--)
            {
                testGradient = network.Layers[k].BackPropagation(testGradient);
            }

            Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
        }

        [Test]
        public void ChainedConvolutionalLayers_PropogateCorrectWeightGradients()
        {
            Vector<double> trainX = Vector<double>.Build.Random(56 * 56 * 3);
            Vector<double> truthY = Vector<double>.Build.Dense(4);
            Vector<double> testWeights = Vector<double>.Build.Random(64 * 6 * 3);

            //note that the number of filters on a convolutional layer turns into the number of channels on the next one
            Network network = new Network(56 * 56 * 3);
            network.Add(new MultiChannelConvolutionalLayer(56 * 56 * 3, kernel: 8, filters: 6, stride: 2, channels: 3));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new MultiChannelConvolutionalLayer(25 * 25 * 6, kernel: 4, filters: 5, stride: 3, channels: 6));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MultiChannelConvolutionalLayer(8 * 8 * 5, kernel: 2, filters: 7, stride: 2, channels: 5));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(4 * 4 * 7, prevFilterCount: 7, poolSize: 2));
            network.Add(new FullyConnectedLayer(10));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(4));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            double networkLossWithWeightAsVariable(Vector<double> x)
            {
                Vector<double>[] splitWeights = MultiChannelConvolutionalLayer.SplitInputToChannels(x, 3, 6 * 64);
                for (int p = 0; p < 3; p++)
                {
                    Vector<double>[] splitWeights2 = MultiChannelConvolutionalLayer.SplitInputToChannels(splitWeights[p], 6, 64);
                    ConvolutionalLayer conv = ((MultiChannelConvolutionalLayer)network.Layers[0]).ChannelOperators[p];
                    for (int k = 0; k < 6; k++)
                    {
                        conv.Weights[k] = MathUtils.Unflatten(splitWeights2[k]);
                    }
                }

                Vector<double> output = network.Predict(trainX);
                return network.Loss(truthY, output);
            }

            Vector<double> finiteDiffWeightGradient = MathUtils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeights);
            List<double> explicitWeightGradientList = new List<double>();

            Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(trainX));

            for (int k = network.Layers.Count - 1; k >= 0; k--)
            {
                outputGradient = network.Layers[k].BackPropagation(outputGradient);
                if (k == 0) // retrieve weight gradient from convolutional layer
                {
                    for (int p = 0; p < 3; p++)
                    {
                        ConvolutionalLayer conv = ((MultiChannelConvolutionalLayer)network.Layers[0]).ChannelOperators[p];
                        for (int y = 0; y < conv.WeightGradients.Length; y++)
                        {
                            Vector<double> weightGrad = MathUtils.Flatten(conv.WeightGradients[y]);
                            for (int q = 0; q < weightGrad.Count; q++)
                                explicitWeightGradientList.Add(weightGrad[q]);
                        }
                    }
                }
            }

            Vector<double> explicitWeightGradient = Vector<double>.Build.DenseOfEnumerable(explicitWeightGradientList);

            Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.0001);
        }
    }
}