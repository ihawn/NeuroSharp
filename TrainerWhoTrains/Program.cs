﻿using MathNet.Numerics.LinearAlgebra;
using NeuroSharp;
using NeuroSharp.Data;
using NeuroSharp.Datatypes;
using NeuroSharp.Enumerations;
using NeuroSharp.Training;

namespace Trainer
{
    public class Trainer
    {
        static void Main(string[] args)
        {
            LetterIdentificationTraining(35);
        }

        static void LetterIdentificationTraining(int epochs)
        {
            ImageDataAggregate data = ImagePreprocessor.GetImageData(
                @"C:\Users\Isaac\Documents\C#\NeuroSharp\Data\english handwritten characters\english.csv",
                @"C:\Users\Isaac\Documents\C#\NeuroSharp\Data\english handwritten characters",
                expectedHeight: 32, expectedWidth: 32, isColor: false
            );

            double trainSplit = 0.9;

            List<Vector<double>> xTrain = data.XValues.Take((int)Math.Floor(trainSplit * data.XValues.Count)).ToList();
            List<Vector<double>> yTrain = data.YValues.Take((int)Math.Floor(trainSplit * data.XValues.Count)).ToList();
            
            List<Vector<double>> xTest = 
                data.XValues.Skip((int)Math.Floor(trainSplit * data.XValues.Count)).Take(Int32.MaxValue).ToList();
            List<Vector<double>> yTest = 
                data.YValues.Skip((int)Math.Floor(trainSplit * data.XValues.Count)).Take(Int32.MaxValue).ToList();

            Network network = new Network(xTrain[0].Count);
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 64, stride: 1, channels: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 32, stride: 2, channels: 64));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 16, stride: 1, channels: 32));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(512));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(128));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(64));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(yTrain[0].Count));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            
            network.Train(xTrain, yTrain, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.002f);
            
            string modelJson = network.SerializeToJSON();
            File.WriteAllText(@"C:\Users\Isaac\Documents\C#\NeuroSharp\NeurosharpBlazorWASM\wwwroot\NetworkModels\characters_model2.json", modelJson);
            
            int i = 0;
            int wrongCount = 0;
            foreach(var test in xTest)
            {
                var output = network.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = yTest[i].ToList().IndexOf(yTest[i].Max());
                Console.WriteLine("Prediction: " + prediction);
                Console.WriteLine("Actual: " + actual + "\n");

                if(prediction != actual)
                    wrongCount++;

                i++;
            }
            double acc = (1d - ((double)wrongCount) / i);
            Console.WriteLine("Accuracy: " + acc);
        }
    }
}