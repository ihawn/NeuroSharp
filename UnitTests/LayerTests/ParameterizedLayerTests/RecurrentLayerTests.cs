using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Training;

namespace UnitTests.LayerTests.ParameterizedLayerTests
{
    internal class RecurrentLayerTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void ForwardPass_ReturnsCorrectResult_Bias0()
        {
            Vector<double> xTest = Vector<double>.Build.DenseOfArray(
                new double[]
                {
                    1, 2, 3, 4, 5, 6, 7, 8
                }
            );

            Network network = new Network(2 * 4);
            network.Add(new RecurrentLayer(2, 4, 3));
            
            ((RecurrentLayer)network.Layers[0]).Weights[(int)RNNWeight.U] = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { -0.04, -0.035, -0.25, 0.22 },
                    { 0.025, 0.69, 0.43, -0.05 },
                    { 0.28, -0.16, -0.3, 0.055 }
                }
            );
            ((RecurrentLayer)network.Layers[0]).Weights[(int)RNNWeight.V] = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { -0.4, -0.3, -0.2 },
                    { 0.2, 0.25, 0.6 },
                    { 0.4, -0.55, 0.2 },
                    { -0.1, 0.3, -0.05 }
                }
            );
            ((RecurrentLayer)network.Layers[0]).Weights[(int)RNNWeight.W] = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { 0.47, -0.35, 0.02 },
                    { 0.15, 0.056, -0.02 },
                    { 0.355, -0.4, -0.254},
                }
            );
            
            Vector<double> pred = network.Predict(xTest);
            Vector<double> expected = Vector<double>.Build.DenseOfArray(
                new double[]
                {
                    -0.18056128, -0.1195259452, -0.6579473, 0.3247899, 0.13280206, -0.4143265177, -0.98280048, 0.40819981992
                }
            );
            
            Assert.IsTrue((pred - expected).L2Norm() < 0.000001);
        }

        [Test]
        public void BackwardPass_ReturnsCorrectInputGradient()
        {
            for (int i = 0; i < 250; i++)
            {
                Random rand = new Random();
                int hiddenSize = rand.Next(1, 25);
                int vocabSize = rand.Next(1, 25);
                int sequenceLength = rand.Next(1, 25);
            
                Vector<double> xTest = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> yTruth = Vector<double>.Build.Random(hiddenSize * vocabSize);

                Network network = new Network(vocabSize * hiddenSize);
                network.Add(new RecurrentLayer(hiddenSize, vocabSize, sequenceLength));
                network.UseLoss(LossType.MeanSquaredError);
            
                double networkLoss(Vector<double> x)
                {
                    x = network.Predict(x);
                    return network.Loss(yTruth, x);
                }

                Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLoss, xTest);
                Vector<double> testGradient = LossFunctions.MeanSquaredErrorPrime(yTruth, network.Predict(xTest));
                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    testGradient = network.Layers[k].BackPropagation(testGradient);
                }

                Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
            }
        }
        
        [Test]
        public void BackwardPass_ReturnsCorrectVGradient()
        {
            for (int i = 0; i < 250; i++)
            {
                Random rand = new Random();
                int hiddenSize = rand.Next(1, 25);
                int vocabSize = rand.Next(1, 25);
                int sequenceLength = rand.Next(1, 25);
                
                Vector<double> xTest = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> yTruth = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> testV = Vector<double>.Build.Random(vocabSize * sequenceLength);

                Network network = new Network(vocabSize * hiddenSize);
                network.Add(new RecurrentLayer(hiddenSize, vocabSize, sequenceLength));
                network.UseLoss(LossType.MeanSquaredError);
            
                double networkLoss(Vector<double> x)
                {
                    RecurrentLayer rec = (RecurrentLayer)network.Layers[0];
                    rec.Weights[(int)RNNWeight.V] = MathUtils.Unflatten(x, vocabSize, sequenceLength);
                    Vector<double> output = network.Predict(xTest);
                    return network.Loss(yTruth, output);
                }

                Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLoss, testV);
                Vector<double> outputGradient = LossFunctions.MeanSquaredErrorPrime(yTruth, network.Predict(xTest));
                outputGradient = network.Layers[0].BackPropagation(outputGradient);
                Vector<double> explicitWeightGradient =
                    MathUtils.Flatten(((RecurrentLayer)network.Layers[0]).WeightGradients[(int)RNNWeight.V]);

                Assert.IsTrue((finiteDiffGradient - explicitWeightGradient).L2Norm() < 0.00001);
            }
        }
        
        [Test]
        public void BackwardPass_ReturnsCorrectUGradient()
        {
            for (int i = 0; i < 500; i++)
            {
                Random rand = new Random();
                int hiddenSize = rand.Next(1, 15);
                int vocabSize = rand.Next(1, 25);
                int sequenceLength = rand.Next(1, 25);
                
                Vector<double> xTest = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> yTruth = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> testU = Vector<double>.Build.Random(sequenceLength * vocabSize);

                Network network = new Network(vocabSize * hiddenSize);
                network.Add(new RecurrentLayer(hiddenSize, vocabSize, sequenceLength));
                network.UseLoss(LossType.MeanSquaredError);
            
                double networkLoss(Vector<double> x)
                {
                    RecurrentLayer rec = (RecurrentLayer)network.Layers[0];
                    rec.Weights[(int)RNNWeight.U] = MathUtils.Unflatten(x, sequenceLength, vocabSize);
                    Vector<double> output = network.Predict(xTest);
                    return network.Loss(yTruth, output);
                }

                Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLoss, testU);
                Vector<double> outputGradient = LossFunctions.MeanSquaredErrorPrime(yTruth, network.Predict(xTest));
                outputGradient = network.Layers[0].BackPropagation(outputGradient);
                Vector<double> explicitWeightGradient =
                    MathUtils.Flatten(((RecurrentLayer)network.Layers[0]).WeightGradients[(int)RNNWeight.U]);

                Assert.IsTrue((finiteDiffGradient - explicitWeightGradient).L2Norm() < 0.001);
            }
        }
        
        [Test]
        public void BackwardPass_ReturnsCorrectWGradient()
        {
            for (int i = 0; i < 500; i++)
            {
                Random rand = new Random();
                int hiddenSize = rand.Next(1, 10);
                int vocabSize = rand.Next(1, 25);
                int sequenceLength = rand.Next(1, 25);
                
                Vector<double> xTest = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> yTruth = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> testW = Vector<double>.Build.Random(sequenceLength * sequenceLength);

                Network network = new Network(vocabSize * hiddenSize);
                network.Add(new RecurrentLayer(hiddenSize, vocabSize, sequenceLength));
                network.UseLoss(LossType.MeanSquaredError);
            
                double networkLoss(Vector<double> x)
                {
                    RecurrentLayer rec = (RecurrentLayer)network.Layers[0];
                    rec.Weights[(int)RNNWeight.W] = MathUtils.Unflatten(x, sequenceLength, sequenceLength);
                    Vector<double> output = network.Predict(xTest);
                    return network.Loss(yTruth, output);
                }

                Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLoss, testW);
                Vector<double> outputGradient = LossFunctions.MeanSquaredErrorPrime(yTruth, network.Predict(xTest));
                outputGradient = network.Layers[0].BackPropagation(outputGradient);
                Vector<double> explicitWeightGradient =
                    MathUtils.Flatten(((RecurrentLayer)network.Layers[0]).WeightGradients[(int)RNNWeight.W]);

                double gradientProximity = (finiteDiffGradient - explicitWeightGradient).L2Norm();
                Assert.IsTrue(gradientProximity < 0.001);
            }
        }

        [Test]
        public void BackwardPass_ReturnsCorrect_bGradient()
        {
            for (int i = 0; i < 250; i++)
            {
                Random rand = new Random();
                int hiddenSize = rand.Next(1, 25);
                int vocabSize = rand.Next(1, 25);
                int sequenceLength = rand.Next(1, 25);
                
                Vector<double> xTest = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> yTruth = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> bTest = Vector<double>.Build.Random(sequenceLength);

                Network network = new Network(vocabSize * hiddenSize);
                network.Add(new RecurrentLayer(hiddenSize, vocabSize, sequenceLength));
                network.UseLoss(LossType.MeanSquaredError);
            
                double networkLoss(Vector<double> x)
                {
                    RecurrentLayer rec = (RecurrentLayer)network.Layers[0];
                    rec.Biases[(int)RNNBias.b] = x;
                    Vector<double> output = network.Predict(xTest);
                    return network.Loss(yTruth, output);
                }

                Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLoss, bTest);
                Vector<double> outputGradient = LossFunctions.MeanSquaredErrorPrime(yTruth, network.Predict(xTest));
                outputGradient = network.Layers[0].BackPropagation(outputGradient);
                Vector<double> explicitBiasGradient = ((RecurrentLayer)network.Layers[0]).BiasGradients[(int)RNNBias.b];

                Assert.IsTrue((finiteDiffGradient - explicitBiasGradient).L2Norm() < 0.00001);
            }
        }
        
        [Test]
        public void BackwardPass_ReturnsCorrect_cGradient()
        {
            for (int i = 0; i < 250; i++)
            {
                Random rand = new Random();
                int hiddenSize = rand.Next(1, 25);
                int vocabSize = rand.Next(1, 25);
                int sequenceLength = rand.Next(1, 25);
                
                Vector<double> xTest = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> yTruth = Vector<double>.Build.Random(hiddenSize * vocabSize);
                Vector<double> cTest = Vector<double>.Build.Random(vocabSize);

                Network network = new Network(vocabSize * hiddenSize);
                network.Add(new RecurrentLayer(hiddenSize, vocabSize, sequenceLength));
                network.UseLoss(LossType.MeanSquaredError);
            
                double networkLoss(Vector<double> x)
                {
                    RecurrentLayer rec = (RecurrentLayer)network.Layers[0];
                    rec.Biases[(int)RNNBias.c] = x;
                    Vector<double> output = network.Predict(xTest);
                    return network.Loss(yTruth, output);
                }

                Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLoss, cTest);
                Vector<double> outputGradient = LossFunctions.MeanSquaredErrorPrime(yTruth, network.Predict(xTest));
                outputGradient = network.Layers[0].BackPropagation(outputGradient);
                Vector<double> explicitBiasGradient = ((RecurrentLayer)network.Layers[0]).BiasGradients[(int)RNNBias.c];

                Assert.IsTrue((finiteDiffGradient - explicitBiasGradient).L2Norm() < 0.00001);
            }
        }
    }
}
