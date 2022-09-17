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
    public class RecurrentNetworkTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void RNN_LearnsCorrectly_WhenSameDataWithDifferentOrder()
        {
            // vocabSize = 3, sequenceLength = 3
            // john eats apples / apples eat john
            List<Vector<double>> xTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(
                    new double[]
                    {
                        1, 0, 0,
                        0, 1, 0,
                        0, 0, 1,
                    }
                ),
                Vector<double>.Build.DenseOfArray(
                    new double[]
                    {
                        0, 0, 1,
                        0, 1, 0,
                        1, 0, 0,
                    }
                )
            };

            List<Vector<double>> yTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(
                    new double [] { 1, 0 }
                ),
                Vector<double>.Build.DenseOfArray(
                    new double [] { 0, 1 }
                )
            };


            Network network = new Network(9);
            network.Add(new RecurrentLayer(3, 3, 16));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(2));
            network.Add(new NeuroSharp.SoftmaxActivationLayer());
            network.UseLoss(LossType.BinaryCrossentropy);
            
            network.Train(xTrain, yTrain, 30, TrainingConfiguration.SGD, OptimizerType.Adam, learningRate: 0.1);

            for (int i = 0; i < xTrain.Count; i++)
            {
                Vector<double> predVector = network.Predict(xTrain[i]);
                int pred = predVector.ToList().IndexOf(predVector.Max());
                int actual = yTrain[i].ToList().IndexOf(yTrain[i].Max());
                
                Console.WriteLine("Prediction: [" + predVector[0] + ", " + predVector[1] + "]");
                Console.WriteLine("Actual: [" + yTrain[i][0] + ", " + yTrain[i][1] + "]");
                Console.WriteLine();
                
                Assert.AreEqual(actual, pred);
            }
        }
        
        [Test]
        public void RNN_LearnsCorrectly_Sequences()
        {
            Random rand = new Random();
            int maxDiff = 1;
            int maxStart = 1;
            int sequenceLength = 5;
            int vocabSize = maxStart + maxDiff * (sequenceLength + 1);
            int trainSize = 500;
            int testSize = 50;

            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();
            
            List<Vector<double>> xTest = new List<Vector<double>>();
            List<Vector<double>> yTest= new List<Vector<double>>();


            for (int i = 0; i < trainSize + testSize; i++)
            {
                int commonDiff = rand.Next(1, maxDiff + 1);
                int start = rand.Next(1, maxStart + 1);

                Vector<double> x = Vector<double>.Build.Dense(sequenceLength);
                Vector<double> y = Vector<double>.Build.Dense(sequenceLength);
                for (int j = 0; j < sequenceLength; j++)
                {
                    x[j] = start + j * commonDiff;
                    y[j] = start + (j + 1) * commonDiff;
                }

                x = DataUtils.OneHotEncodeVector(x, vocabSize);
                y = DataUtils.OneHotEncodeVector(y, vocabSize);

                if (i < trainSize)
                {
                    xTrain.Add(x);
                    yTrain.Add(y);
                }
                else
                {
                    xTest.Add(x);
                    yTest.Add(y);
                }
            }
            

            Network network = new Network(sequenceLength * vocabSize);
            network.Add(new RecurrentLayer(sequenceLength, vocabSize, 80));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            network.Train(xTrain, yTrain, 30, TrainingConfiguration.SGD, OptimizerType.Adam, learningRate: 0.1);

            for (int i = 0; i < xTest.Count; i++)
            {
                Vector<double> predVector = network.Predict(xTest[i]);
                Vector<double> encodedLastNumberPred = predVector.SubVector((sequenceLength - 1) * vocabSize, vocabSize);
                Vector<double> encodedLastNumberActual = yTest[i].SubVector((sequenceLength - 1) * vocabSize, vocabSize);
                int predNumber = encodedLastNumberPred.ToList().IndexOf(encodedLastNumberPred.Max()) + 1;
                int actualNumber = encodedLastNumberActual.ToList().IndexOf(encodedLastNumberActual.Max()) + 1;
                
                Console.WriteLine("Prediction: " + predNumber);
                Console.WriteLine("Actual: " + actualNumber);
                Console.WriteLine();
            }
        }
    }
}