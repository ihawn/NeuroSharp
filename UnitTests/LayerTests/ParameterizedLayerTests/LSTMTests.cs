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
        public void LSTM_BackPropagation_ReturnsCorrectInputGradient()
        {
            for (int i = 1; i <= 10; i++)
            {
                for (int j = 1; j <= 10; j++)
                {
                    for (int n = 1; n <= 10; n++)
                    {
                        int vocabSize = i;
                        int sequenceLength = j;
                        int hiddenSize = n;
                        bool bidirectional = false; //new Random().NextDouble() > 0.5;

                        Vector<double> truthY = Vector<double>.Build.Random(vocabSize * (bidirectional ? 2 : 1));
                        Vector<double> testX = Vector<double>.Build.Random(sequenceLength * vocabSize);

                        Network network = new Network(27 * 12);
                        network.Add(new LSTMLayer(vocabSize, hiddenSize, sequenceLength, bidirectional));
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
        }



        [Test]
        public void LSTM_BackPropagation_ReturnsCorrect_Gate_Weight_Gradient()
        {
            for (int lstmGateId = 0; lstmGateId < 4; lstmGateId++)
            {
                for (int i = 1; i <= 10; i++)
                {
                    for (int j = 1; j <= 10; j++)
                    {
                        for (int n = 1; n <= 10; n++)
                        {
                            int vocabSize = i;
                            int sequenceLength = j;
                            int hiddenSize = n;
                            bool bidirectional = false; //new Random().NextDouble() > 0.5;

                            Vector<double> truthY = Vector<double>.Build.Random(vocabSize * (bidirectional ? 2 : 1));
                            Vector<double> testX = Vector<double>.Build.Random(sequenceLength * vocabSize);
                            Vector<double> testWeight =
                                Vector<double>.Build.Random((vocabSize + hiddenSize) * hiddenSize);

                            Network network = new Network(27 * 12);
                            network.Add(new LSTMLayer(vocabSize, hiddenSize, sequenceLength, bidirectional));
                            network.Add(new ActivationLayer(ActivationType.Tanh));
                            network.Add(new SoftmaxActivationLayer());
                            network.UseLoss(LossType.CategoricalCrossentropy);

                            double networkLossHWeightInput(Vector<double> x)
                            {
                                LSTMLayer lstm = (LSTMLayer)network.Layers[0];
                                lstm.LSTMGates[(int)LSTMParameter.F].Weights[0] =
                                    MathUtils.Unflatten(x, vocabSize + hiddenSize, hiddenSize);
                                x = network.Predict(testX);
                                return network.Loss(truthY, x);
                            }

                            Vector<double> finiteDiffGradient =
                                MathUtils.FiniteDifferencesGradient(networkLossHWeightInput, testWeight);
                            Vector<double> testGradient =
                                LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
                            Vector<double> explicitWeightGradient = null;
                            for (int k = network.Layers.Count - 1; k >= 0; k--)
                            {
                                testGradient = network.Layers[k].BackPropagation(testGradient);
                                if (k == 0)
                                {
                                    LSTMLayer lstm = (LSTMLayer)network.Layers[0];
                                    explicitWeightGradient = MathUtils.Flatten(lstm.LSTMGates[(int)LSTMParameter.F]
                                        .WeightGradients[0].Transpose());
                                }
                            }

                            Assert.IsTrue((finiteDiffGradient - explicitWeightGradient).L2Norm() < 0.00001);
                        }
                    }
                }
            }
        }
        
        [Test]
        public void LSTM_BackPropagation_ReturnsCorrect_Gate_Bias_Gradient()
        {
            for (int lstmGateId = 0; lstmGateId < 4; lstmGateId++)
            {
                for (int i = 1; i <= 10; i++)
                {
                    for (int j = 1; j <= 10; j++)
                    {
                        for (int n = 1; n <= 10; n++)
                        {
                            int vocabSize = i;
                            int sequenceLength = j;
                            int hiddenSize = n;
                            bool bidirectional = false; //new Random().NextDouble() > 0.5;

                            Vector<double> truthY = Vector<double>.Build.Random(vocabSize * (bidirectional ? 2 : 1));
                            Vector<double> testX = Vector<double>.Build.Random(sequenceLength * vocabSize);
                            Vector<double> testBias = Vector<double>.Build.Random(hiddenSize);

                            Network network = new Network(27 * 12);
                            network.Add(new LSTMLayer(vocabSize, hiddenSize, sequenceLength, bidirectional));
                            network.Add(new ActivationLayer(ActivationType.Tanh));
                            network.Add(new SoftmaxActivationLayer());
                            network.UseLoss(LossType.CategoricalCrossentropy);

                            double networkLossHWeightInput(Vector<double> x)
                            {
                                LSTMLayer lstm = (LSTMLayer)network.Layers[0];
                                lstm.LSTMGates[lstmGateId].Biases[0] = x;
                                x = network.Predict(testX);
                                return network.Loss(truthY, x);
                            }

                            Vector<double> finiteDiffGradient =
                                MathUtils.FiniteDifferencesGradient(networkLossHWeightInput, testBias, h: 0.0000001);
                            Vector<double> testGradient =
                                LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
                            Vector<double> explicitWeightGradient = null;
                            for (int k = network.Layers.Count - 1; k >= 0; k--)
                            {
                                testGradient = network.Layers[k].BackPropagation(testGradient);
                                if (k == 0)
                                {
                                    LSTMLayer lstm = (LSTMLayer)network.Layers[0];
                                    explicitWeightGradient = lstm.LSTMGates[lstmGateId].BiasGradients[0];
                                }
                            }

                            Assert.IsTrue((finiteDiffGradient - explicitWeightGradient).L2Norm() < 0.00001);
                        }
                    }
                }
            }
        }
    }
}

//todo train batching by stacking vector batches into a matrix