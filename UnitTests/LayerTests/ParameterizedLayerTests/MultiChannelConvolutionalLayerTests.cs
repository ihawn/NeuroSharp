using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Data;
using NeuroSharp.Training;

namespace UnitTests
{
    public class MultiChannelConvolutionalLayerTests
    {
        [SetUp]
        public void Setup()
        {
        }

        #region Operator Tests
        [Test]
        public void SplitInputToChannels_ReturnsCorrectChannelSplit()
        {
            Vector<double> input1 = Vector<double>.Build.DenseOfArray(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
            Vector<double>[] output1 = MultiChannelConvolutionalLayer.SplitInputToChannels(input1, 3, 5);
            Vector<double>[] expected1 = new Vector<double>[]
            {
                Vector<double>.Build.DenseOfArray(new double[] {1, 2, 3, 4, 5}),
                Vector<double>.Build.DenseOfArray(new double[] {6, 7, 8, 9, 10}),
                Vector<double>.Build.DenseOfArray(new double[] {11, 12, 13, 14, 15}),
            };

            Assert.AreEqual(expected1, output1);
        }  

        [Test]
        public void CombineChannelBackpropagation_ReturnsCorrectVectorCombination()
        {
            Vector<double>[] input1 = new Vector<double>[]
            {
                Vector<double>.Build.DenseOfArray(new double[] {1, 2, 3, 4, 5}),
                Vector<double>.Build.DenseOfArray(new double[] {6, 7, 8, 9, 10}),
                Vector<double>.Build.DenseOfArray(new double[] {11, 12, 13, 14, 15}),
            };
            Vector<double> expected1 = Vector<double>.Build.DenseOfArray(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
            Vector<double> output1 = MultiChannelConvolutionalLayer.CombineChannelBackPropagation(input1, 3, 5);

            Assert.AreEqual(expected1, output1);
        }
        #endregion

        #region Propagation Tests
        [Test]
        public void MultiChannelConvLayer_ForwardPropagation_ReturnsCorrectShape()
        {
            //setup 1
            Vector<double> input1 = Vector<double>.Build.DenseOfArray(
            new double[]
            {
                1, 2, 3, 4
            });

            MultiChannelConvolutionalLayer layer1 = new MultiChannelConvolutionalLayer(inputSize: 4, kernel: 2, filters: 1, stride: 1, channels: 1);
            Vector<double> output1 = layer1.ForwardPropagation(input1);

            //setup 2
            Vector<double> input2 = Vector<double>.Build.DenseOfArray(
            new double[]
            {
                1, 2, 3, 4,   //red
                5, 6, 7, 8,   //green
                9, 10, 11, 12 // blue
            });

            MultiChannelConvolutionalLayer layer2 = new MultiChannelConvolutionalLayer(inputSize: 12, kernel: 2, filters: 1, stride: 1, channels: 3);
            Vector<double> output2 = layer2.ForwardPropagation(input2);

            //setup 3
            Vector<double> input3 = Vector<double>.Build.DenseOfArray(
            new double[]
            {
                1, 2, 3, 4,   //red
                5, 6, 7, 8,   //green
                9, 10, 11, 12 // blue
            });

            MultiChannelConvolutionalLayer layer3 = new MultiChannelConvolutionalLayer(inputSize: 12, kernel: 2, filters: 7, stride: 1, channels: 3);
            Vector<double> output3 = layer3.ForwardPropagation(input3);

            //setup 4
            Vector<double> input4 = Vector<double>.Build.DenseOfArray(
            new double[]
            {
                1, 2, 3, 4, 13, 14, 15, 16, 17,     //red
                5, 6, 7, 8, 18, 19, 20, 21, 22,     //green
                9, 10, 11, 12, 23, 24, 25, 26, 27   // blue
            });

            MultiChannelConvolutionalLayer layer4 = new MultiChannelConvolutionalLayer(inputSize: 27, kernel: 2, filters: 7, stride: 1, channels: 3);
            Vector<double> output4 = layer4.ForwardPropagation(input4);


            Assert.AreEqual(1, output1.Count);
            Assert.AreEqual(1, output2.Count);
            Assert.AreEqual(7, output3.Count);
            Assert.AreEqual(28, output4.Count);
        }

        [Test]
        public void MultiChannelConvLayer_BackPropagation_ReturnsCorrectInputGradient_CategoricalCrossentropy_Stride1()
        {
            for(int s = 0; s < 25; s++)
            {
                Vector<double> testX = Vector<double>.Build.Random(27);
                Vector<double> truthY = Vector<double>.Build.Random(28);

                Network network = new Network();
                network.Add(new MultiChannelConvolutionalLayer(27, kernel: 2, filters: 7, stride: 1, channels: 3));
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
        public void MultiChannelConvLayer_BackPropagation_ReturnsCorrectInputGradient_CategoricalCrossentropy_Stride2()
        {
            for (int s = 0; s < 25; s++)
            {
                Vector<double> testX = Vector<double>.Build.Random(48);
                Vector<double> truthY = Vector<double>.Build.Random(28);

                Network network = new Network();
                network.Add(new MultiChannelConvolutionalLayer(48, kernel: 2, filters: 7, stride: 2, channels: 3));
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
        public void MultiChannelConvLayer_BackPropagation_ComputesCorrectWeightGradient_CategoricalCrossentropy_Stride1()
        {
            for (int s = 0; s < 25; s++)
            {
                Vector<double> testX = Vector<double>.Build.Random(36);
                Vector<double> truthY = Vector<double>.Build.Random(13 * 4);
                Vector<double> testWeights = Vector<double>.Build.Random(4 * 13 * 4);

                Network network = new Network();
                network.Add(new MultiChannelConvolutionalLayer(inputSize: 36, kernel: 2, filters: 13, stride: 1, channels: 4));
                network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                double networkLossWithWeightAsVariable(Vector<double> x)
                {
                    Vector<double>[] splitWeights = MultiChannelConvolutionalLayer.SplitInputToChannels(x, 4, 13 * 4);
                    for (int p = 0; p < 4; p++)
                    {
                        Vector<double>[] splitWeights2 = MultiChannelConvolutionalLayer.SplitInputToChannels(splitWeights[p], 13, 4);
                        ConvolutionalLayer conv = ((MultiChannelConvolutionalLayer)network.Layers[0]).ChannelOperators[p];
                        for (int k = 0; k < 13; k++)
                        {
                            conv.Weights[k] = Utils.Unflatten(splitWeights2[k]);
                        }
                    }

                    Vector<double> output = network.Predict(testX);
                    return network.Loss(truthY, output);
                }

                Vector<double> finiteDiffWeightGradient = Utils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeights);
                List<double> explicitWeightGradientList = new List<double>();

                Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    outputGradient = network.Layers[k].BackPropagation(outputGradient);
                    if (k == 0) // retrieve weight gradient from convolutional layer
                    {
                        for (int p = 0; p < 4; p++)
                        {
                            ConvolutionalLayer conv = ((MultiChannelConvolutionalLayer)network.Layers[0]).ChannelOperators[p];
                            for (int y = 0; y < conv.WeightGradients.Length; y++)
                            {
                                Vector<double> weightGrad = Utils.Flatten(conv.WeightGradients[y]);
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

        [Test]
        public void MultiChannelConvLayer_BackPropagation_ComputesCorrectWeightGradient_CategoricalCrossentropy_Stride2()
        {
            for(int s = 0; s < 25; s++)
            {
                Vector<double> testX = Vector<double>.Build.Random(80);
                Vector<double> truthY = Vector<double>.Build.Random(13 * 4);
                Vector<double> testWeights = Vector<double>.Build.Random(5 * 13 * 4);

                Network network = new Network();
                network.Add(new MultiChannelConvolutionalLayer(inputSize: 80, kernel: 2, filters: 13, stride: 2, channels: 5));
                network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

                double networkLossWithWeightAsVariable(Vector<double> x)
                {
                    Vector<double>[] splitWeights = MultiChannelConvolutionalLayer.SplitInputToChannels(x, 5, 13 * 4);
                    for (int p = 0; p < 5; p++)
                    {
                        Vector<double>[] splitWeights2 = MultiChannelConvolutionalLayer.SplitInputToChannels(splitWeights[p], 13, 4);
                        ConvolutionalLayer conv = ((MultiChannelConvolutionalLayer)network.Layers[0]).ChannelOperators[p];
                        for (int k = 0; k < 13; k++)
                        {
                            conv.Weights[k] = Utils.Unflatten(splitWeights2[k]);
                        }
                    }

                    Vector<double> output = network.Predict(testX);
                    return network.Loss(truthY, output);
                }

                Vector<double> finiteDiffWeightGradient = Utils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeights);
                List<double> explicitWeightGradientList = new List<double>();

                Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    outputGradient = network.Layers[k].BackPropagation(outputGradient);
                    if (k == 0) // retrieve weight gradient from convolutional layer
                    {
                        for (int p = 0; p < 5; p++)
                        {
                            ConvolutionalLayer conv = ((MultiChannelConvolutionalLayer)network.Layers[0]).ChannelOperators[p];
                            for (int y = 0; y < conv.WeightGradients.Length; y++)
                            {
                                Vector<double> weightGrad = Utils.Flatten(conv.WeightGradients[y]);
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
        #endregion
    }
}