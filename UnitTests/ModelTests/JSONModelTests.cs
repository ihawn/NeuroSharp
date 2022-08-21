﻿using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Training;
using Newtonsoft.Json;

namespace UnitTests.ModelTests
{
    public class JSONModelTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void SerializeToJSON_NetworkSerializesToJsonAndDeserializes_WithDenseLayers()
        {
            double[][] xx =
            {
                new double[]{ 0, 0 },
                new double[]{ 0, 1 },
                new double[]{ 1, 0 },
                new double[]{ 1, 1 }
            };
            List<Vector<double>> xTrain = new List<Vector<double>>();
            foreach (var x in xx)
                xTrain.Add(Vector<double>.Build.DenseOfArray(x));

            double[][] yy =
            {
                new double[]{ 0 },
                new double[]{ 1 },
                new double[]{ 1 },
                new double[]{ 0 }
            };
            List<Vector<double>> yTrain = new List<Vector<double>>();
            foreach (var y in yy)
                yTrain.Add(Vector<double>.Build.DenseOfArray(y));


            Network network = new Network();
            network.Add(new FullyConnectedLayer(2, 3));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(3, 1));
            network.UseLoss(LossType.MeanSquaredError);

            network.SGDTrain(xTrain, yTrain, epochs: 1000, optimizerType: OptimizerType.GradientDescent, learningRate: 0.1f);
            string modelJson = network.SerializeToJSON();
            Network deserializedNetwork = Network.DeserializeNetworkJSON(modelJson);

            foreach (var test in xTrain)
            {
                Vector<double> output1 = network.Predict(test);
                Vector<double> output2 = deserializedNetwork.Predict(test);

                Assert.AreEqual(output1, output2);
            }        
        }
        
        [Test]
        public void SerializeToJSON_NetworkSerializesToJsonAndDeserializes_WithDenseAndConvolutional()
        {
            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();

            for (int i = 0; i < 15; i++)
            {
                xTrain.Add(Vector<double>.Build.Random(28 * 28));
                yTrain.Add(Vector<double>.Build.Random(10));
            }

            Network network = new Network();
            network.Add(new MultiChannelConvolutionalLayer(28 * 28, kernel: 3, filters: 8, channels: 1, stride: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MultiChannelConvolutionalLayer(26 * 26 * 8, kernel: 3, filters: 2, channels: 8, stride: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MultiChannelConvolutionalLayer(24 * 24 * 2, kernel: 2, filters: 2, channels: 2, stride: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(23 * 23 * 2, prevFilterCount: 2, poolSize: 2));
            network.Add(new FullyConnectedLayer(22 * 22 * 2, 128));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(128, 10));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            network.MinibatchTrain(xTrain, yTrain, epochs: 5, OptimizerType.Adam, batchSize: 64, learningRate: 0.001f);
            string modelJson = network.SerializeToJSON();
            Network deserializedNetwork = Network.DeserializeNetworkJSON(modelJson);

            foreach (Vector<double> x in xTrain)
            {
                Vector<double> pred1 = network.Predict(x);
                Vector<double> pred2 = deserializedNetwork.Predict(x);
                Assert.AreEqual(pred1, pred2);
            }
        }
        
        //todo: write tests for continuing training after deserializing a model
    }
}