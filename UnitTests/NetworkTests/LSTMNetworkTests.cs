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
    public class LSTMNetworkTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void ChainedConvolutionalLayers_PropogateCorrectInputGradients_CategoricalCrossentropy()
        {
           
        }

        [Test]
        public void LSTMNetwork_TrainsCorrectly()
        {
            List<List<int>> xx = new List<List<int>>()
            {
                new List<int> { 2, 4, 6, 8},
                new List<int> { 0, 2, 4, 6},
                new List<int> { 6, 8, 10, 12},
                new List<int> { 8, 10, 12, 14}
            };
            List<int> yy = new List<int> { 10, 8, 14, 16 };

            int vocabSize = 176;
            int sequenceLength = 4;

            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();

            for (int i = 0; i < xx.Count; i++)
            {
                Vector<double> x = Vector<double>.Build.Dense(vocabSize * sequenceLength);
                for (int j = 0; j < xx[i].Count; j++)
                {
                    x[j * vocabSize + xx[i][j]] = 1;
                }
                xTrain.Add(x);
                
                Vector<double> y = Vector<double>.Build.Dense(vocabSize);
                y[yy[i]] = 1;
                yTrain.Add(y);
            }
            
            
            Network network = new Network(vocabSize * sequenceLength);
            network.Add(new LSTMLayer(vocabSize, 16, sequenceLength));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(vocabSize));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.BinaryCrossentropy);
            
            network.Train(xTrain, yTrain, 100, TrainingConfiguration.SGD, OptimizerType.GradientDescent, learningRate: 0.03);
            
            int wrongCount = 0;
            for (int i = 0; i < xTrain.Count; i++)
            {
                Vector<double> pred = network.Predict(xTrain[i]);
                int predNum = pred.ToList().IndexOf(pred.Max()) + 1;
                int actualNum = yTrain[i].ToList().IndexOf(1) + 1;
                Console.WriteLine("Prediction: " + predNum);
                Console.WriteLine("Actual: " + actualNum);

                if (predNum != actualNum)
                    wrongCount++;
            }

            double perc = 1 - (double)wrongCount / xTrain.Count;
            Assert.AreEqual(perc, 1);
        }
    }
}