using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.MathUtils;
using NeuroSharp.Data;

namespace NeuroSharp
{
    public class NeuroSharp
    {
        static void Main(string[] args)
        {
            Mnist_Test();
            //XOR_Test();
        }

        static void XOR_Test()
        {
            float[][] xx =
{
                new float[]{ 0, 0 },
                new float[]{ 0, 1 },
                new float[]{ 1, 0 },
                new float[]{ 1, 1 }
            };
            List<Vector<float>> xTrain = new List<Vector<float>>();
            foreach (var x in xx)
                xTrain.Add(Vector<float>.Build.DenseOfArray(x));

            float[][] yy =
            {
                new float[]{ 0 },
                new float[]{ 1 },
                new float[]{ 1 },
                new float[]{ 0 }
            };
            List<Vector<float>> yTrain = new List<Vector<float>>();
            foreach (var y in yy)
                yTrain.Add(Vector<float>.Build.DenseOfArray(y));


            Network network = new Network();
            network.Add(new FullyConnectedLayer(2, 3));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(3, 1));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));

            //train
            network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);
            network.BatchTrain(xTrain, yTrain, epochs: 1000, learningRate: 0.1f);

            //test
            var output = network.Predict(xTrain);
            foreach (var o in output)
                Console.WriteLine((float)o[0]);
        }

        static void Mnist_Test()
        {
            //training data
            List<Vector<float>> xTrain = new List<Vector<float>>();
            List<Vector<float>> yTrain = new List<Vector<float>>();

            var trainData = MnistReader.ReadTrainingData().ToList();
            for(int n = 0; n < 20000;/* trainData.Count;*/ n++)
            {
                var image = trainData[n];

                float[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t/256f).ToArray();
                xTrain.Add(Vector<float>.Build.DenseOfArray(flattenedNormalized));

                float[] categorical = new float[10];
                categorical[image.Label] = 1;
                yTrain.Add(Vector<float>.Build.DenseOfArray(categorical));
            }


            //testing data
            List<Vector<float>> xTest = new List<Vector<float>>();
            List<Vector<float>> yTest = new List<Vector<float>>();

            var testData = MnistReader.ReadTestData().ToList();
            for (int n = 0; n < testData.Count; n++)
            {
                var image = testData[n];

                float[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t / 256f).ToArray();
                xTest.Add(Vector<float>.Build.DenseOfArray(flattenedNormalized));

                float[] categorical = new float[10];
                categorical[image.Label] = 1;
                yTest.Add(Vector<float>.Build.DenseOfArray(categorical));
            }

            //build network
            Network network = new Network();
            network.Add(new FullyConnectedLayer(28 * 28, 100));
           //network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(100, 50));
            //network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(50, 10));
            //network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

            //train
            network.AdamTrain(xTrain, yTrain, epochs: 15);

            //test
            int i = 0;
            int wrongCount = 0;
            foreach(var test in xTest)
            {
                var output = network.Predict(new List<Vector<float>>() { test });
                int prediction = output[0].ToList().IndexOf(output[0].Max());
                int actual = yTest[i].ToList().IndexOf(yTest[i].Max());
                Console.WriteLine("Prediction: " + prediction);
                Console.WriteLine("Actual: " + actual + "\n");

                if(prediction != actual)
                    wrongCount++;

                i++;
            }
            Console.WriteLine("Accuracy: " + (1f - ((float)wrongCount)/((float)i)));
        }
    }
}
