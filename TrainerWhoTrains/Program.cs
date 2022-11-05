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
            LetterIdentificationTraining(5);
        }

        static void LetterIdentificationTraining(int epochs)
        {
            ImageDataAggregate data = ImagePreprocessor.GetImageData(
                @"C:\Users\Isaac\Documents\C#\NeuroSharp\Data\english handwritten characters\english.csv",
                @"C:\Users\Isaac\Documents\C#\NeuroSharp\Data\english handwritten characters",
                expectedHeight: 64, expectedWidth: 64, isColor: false
            );

            Network network = new Network(data.XValues[0].Count);
            network.Add(new ConvolutionalLayer(kernel: 4, filters: 8, stride: 2, channels: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 4, stride: 2, channels: 16));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(128));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(64));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(data.YValues[0].Count));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            
            network.Train(data.XValues, data.YValues, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.002f);
        }
    }
}