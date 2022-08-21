using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Training;

namespace UnitTests
{
    public class ConvolutionLayerTests
    {
        [SetUp]
        public void Setup()
        {
        }

        #region Convolutional Gradient Tests For 1 Filter
        //Categorical Crossentropy Loss
        [Test]
        public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_KernelSameSizeAsImageSoNoStride()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 25; i++)
                squares.Add((int)Math.Pow(i, 2));
            foreach (int i in squares)
            {
                Vector<double> truthY = Vector<double>.Build.Random(1);
                Vector<double> testX = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(i), filters: 1, stride: 1));
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

        [Test]
        public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_WithStride1()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    int outdim = (int)Math.Floor(Math.Sqrt(i) - Math.Sqrt(j)) + 1;
                    Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim);
                    Vector<double> testX = Vector<double>.Build.Random(i);

                    Network network = new Network();
                    network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: 1, stride: 1));
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

        [Test]
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_KernelSameSizeAsImageSoNoStride()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));
            foreach (int i in squares)
            {
                Vector<double> truthY = Vector<double>.Build.Random(1);
                Vector<double> testX = Vector<double>.Build.Random(i);
                Vector<double> testWeight = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(i), filters: 1, stride: 1));
                network.Add(new ActivationLayer(ActivationType.Tanh));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossType.CategoricalCrossentropy);

                double networkLossWithWeightAsVariable(Vector<double> x)
                {
                    ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                    conv.Weights[0] = MathUtils.Unflatten(x);
                    Vector<double> output = network.Predict(testX);
                    return network.Loss(truthY, output);
                }

                Vector<double> finiteDiffWeightGradient = MathUtils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeight);
                Vector<double> explicitWeightGradient = Vector<double>.Build.Dense(i);

                Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    outputGradient = network.Layers[k].BackPropagation(outputGradient);
                    if (k == 0) // retrieve weight gradient from convolutional layer
                        explicitWeightGradient = MathUtils.Flatten(((ConvolutionalLayer)network.Layers[0]).WeightGradients[0]);
                }

                Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.00001);
            }
        }

        [Test]
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_WithStride1()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    int outdim = (int)Math.Floor(Math.Sqrt(i) - Math.Sqrt(j)) + 1;
                    Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim);
                    Vector<double> testX = Vector<double>.Build.Random(i);
                    Vector<double> testWeight = Vector<double>.Build.Random(j);

                    Network network = new Network();
                    network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: 1, stride: 1));
                    network.Add(new ActivationLayer(ActivationType.Tanh));
                    network.Add(new SoftmaxActivationLayer());
                    network.UseLoss(LossType.CategoricalCrossentropy);

                    double networkLossWithWeightAsVariable(Vector<double> x)
                    {
                        ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                        conv.Weights[0] = MathUtils.Unflatten(x);
                        Vector<double> output = network.Predict(testX);
                        return network.Loss(truthY, output);
                    }

                    Vector<double> finiteDiffWeightGradient = MathUtils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeight);
                    Vector<double> explicitWeightGradient = Vector<double>.Build.Dense(j);

                    Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                    for (int k = network.Layers.Count - 1; k >= 0; k--)
                    {
                        outputGradient = network.Layers[k].BackPropagation(outputGradient);
                        if (k == 0) // retrieve weight gradient from convolutional layer
                            explicitWeightGradient = MathUtils.Flatten(((ConvolutionalLayer)network.Layers[0]).WeightGradients[0]);
                    }

                    Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.0001);
                }
            }
        }

        //TODO
        //Mean Squared Error Loss
       /*  [Test]
         public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithMSESoftmaxTanhAndConvolutionalLayer_KernelSameSizeAsImageSoNoStride()
         {
             List<int> squares = new List<int>();
             for (int i = 1; i < 25; i++)
                 squares.Add((int)Math.Pow(i, 2));
             foreach (int i in squares)
             {

             }
             Vector<double> truthY = Vector<double>.Build.Random(1);
             Vector<double> testX = Vector<double>.Build.Random(4);

             Network network = new Network();
             network.Add(new ConvolutionalLayer(4, kernel: (int)Math.Sqrt(4), filters: 1, stride: 1));
             network.Add(new ActivationLayer(ActivationType.Tanh));
             network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

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
         }*/

        /*[Test]
        public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithMSESoftmaxTanhAndConvolutionalLayer_WithStride1()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    int outdim = (int)Math.Floor(Math.Sqrt(i) - Math.Sqrt(j)) + 1;
                    Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim);
                    Vector<double> testX = Vector<double>.Build.Random(i);

                    Network network = new Network();
                    network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: 1, stride: 1));
                    network.Add(new ActivationLayer(ActivationType.Tanh));
                    network.Add(new SoftmaxActivationLayer());
                    network.UseLoss(LossType.CategoricalCrossentropy);

                    double networkLoss(Vector<double> x)
                    {
                        x = network.Predict(x);
                        return network.Loss(truthY, x);
                    }

                    Vector<double> finiteDiffGradient = Utils.FiniteDifferencesGradient(networkLoss, testX);
                    Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
                    for (int k = network.Layers.Count - 1; k >= 0; k--)
                    {
                        testGradient = network.Layers[k].BackPropagation(testGradient, OptimizerType.Adam, 1, 0.0001);
                    }

                    Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
                }
            }
        }

        [Test]
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithMSESoftmaxTanhAndConvolutionalLayer_KernelSameSizeAsImageSoNoStride()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));
            foreach (int i in squares)
            {
                Vector<double> truthY = Vector<double>.Build.Random(1);
                Vector<double> testX = Vector<double>.Build.Random(i);
                Vector<double> testWeight = Vector<double>.Build.Random(i);

                Network network = new Network();
                network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(i), filters: 1, stride: 1));
                network.Add(new ActivationLayer(ActivationType.Tanh));
                network.Add(new SoftmaxActivationLayer());
                network.UseLoss(LossType.CategoricalCrossentropy);

                double networkLossWithWeightAsVariable(Vector<double> x)
                {
                    ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                    conv.Weights[0] = Utils.Unflatten(x);
                    Vector<double> output = network.Predict(testX);
                    return network.Loss(truthY, output);
                }

                Vector<double> finiteDiffWeightGradient = Utils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeight);
                Vector<double> explicitWeightGradient = Vector<double>.Build.Dense(i);

                Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                for (int k = network.Layers.Count - 1; k >= 0; k--)
                {
                    outputGradient = network.Layers[k].BackPropagation(outputGradient, OptimizerType.Adam, 1, 0.0001);
                    if (k == 0) // retrieve weight gradient from convolutional layer
                        explicitWeightGradient = Utils.Flatten(((ConvolutionalLayer)network.Layers[0]).WeightGradient[0]);
                }

                Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.00001);
            }
        }

        [Test]
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithMSESoftmaxTanhAndConvolutionalLayer_WithStride1()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 10; i++)
                squares.Add((int)Math.Pow(i, 2));

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    int outdim = (int)Math.Floor(Math.Sqrt(i) - Math.Sqrt(j)) + 1;
                    Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim);
                    Vector<double> testX = Vector<double>.Build.Random(i);
                    Vector<double> testWeight = Vector<double>.Build.Random(j);

                    Network network = new Network();
                    network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: 1, stride: 1));
                    network.Add(new ActivationLayer(ActivationType.Tanh));
                    network.Add(new SoftmaxActivationLayer());
                    network.UseLoss(LossType.CategoricalCrossentropy);

                    double networkLossWithWeightAsVariable(Vector<double> x)
                    {
                        ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                        conv.Weights[0] = Utils.Unflatten(x);
                        Vector<double> output = network.Predict(testX);
                        return network.Loss(truthY, output);
                    }

                    Vector<double> finiteDiffWeightGradient = Utils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeight);
                    Vector<double> explicitWeightGradient = Vector<double>.Build.Dense(j);

                    Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                    for (int k = network.Layers.Count - 1; k >= 0; k--)
                    {
                        outputGradient = network.Layers[k].BackPropagation(outputGradient, OptimizerType.Adam, 1, 0.0001);
                        if (k == 0) // retrieve weight gradient from convolutional layer
                            explicitWeightGradient = Utils.Flatten(((ConvolutionalLayer)network.Layers[0]).WeightGradient[0]);
                    }

                    Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.0001);
                }
            }
        }*/
        #endregion

        #region Stride >= 1 Tests
        //Stride >= 1 (but multiple of input)
        [Test]
        public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_WithStrideMoreThan1()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 12; i++)
                squares.Add((int)Math.Pow(i, 2));
            List<int> nums = new List<int>();
            for (int i = 1; i < 16; i++)
                nums.Add(i);

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    foreach (int s in nums.Where(x => Math.Abs(Math.Floor((Math.Sqrt(i) - Math.Sqrt(j)) / x) - ((Math.Sqrt(i) - Math.Sqrt(j)) / x)) < 0.0000001)) // where stride goes into the image evenly
                    {
                        int outdim = (int)Math.Floor((Math.Sqrt(i) - Math.Sqrt(j)) / s) + 1;
                        Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim);
                        Vector<double> testX = Vector<double>.Build.Random(i);

                        Network network = new Network();
                        network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: 1, stride: s));
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
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_WithStrideMoreThan1()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 16; i++)
                squares.Add((int)Math.Pow(i, 2));
            List<int> nums = new List<int>();
            for(int i = 0; i < 16; i++)
                nums.Add(i);

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    foreach(int s in nums.Where(x => Math.Abs(Math.Floor((Math.Sqrt(i) - Math.Sqrt(j)) / x) - ((Math.Sqrt(i) - Math.Sqrt(j)) / x)) < 0.0000001)) // where stride goes into the image evenly
                    {
                        int outdim = (int)Math.Floor((Math.Sqrt(i) - Math.Sqrt(j)) / s) + 1;
                        Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim);
                        Vector<double> testX = Vector<double>.Build.Random(i);
                        Vector<double> testWeight = Vector<double>.Build.Random(j);

                        Network network = new Network();
                        network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: 1, stride: s));
                        network.Add(new ActivationLayer(ActivationType.Tanh));
                        network.Add(new SoftmaxActivationLayer());
                        network.UseLoss(LossType.CategoricalCrossentropy);

                        double networkLossWithWeightAsVariable(Vector<double> x)
                        {
                            ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                            conv.Weights[0] = MathUtils.Unflatten(x);
                            Vector<double> output = network.Predict(testX);
                            return network.Loss(truthY, output);
                        }

                        Vector<double> finiteDiffWeightGradient = MathUtils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeight);
                        Vector<double> explicitWeightGradient = Vector<double>.Build.Dense(j);

                        Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                        for (int k = network.Layers.Count - 1; k >= 0; k--)
                        {
                            outputGradient = network.Layers[k].BackPropagation(outputGradient);
                            if (k == 0) // retrieve weight gradient from convolutional layer
                                explicitWeightGradient = MathUtils.Flatten(((ConvolutionalLayer)network.Layers[0]).WeightGradients[0]);
                        }

                        Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.0001);
                    }
                }
            }
        }
        #endregion

        #region Convolutional Gradient Tests For Multiple Filters
        [Test]
        public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_WithStride1_MultipleFilters()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 8; i++)
                squares.Add((int)Math.Pow(i, 2));

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    for(int n = 1; n < 16; n++) // filter count
                    {
                        int outdim = (int)Math.Floor(Math.Sqrt(i) - Math.Sqrt(j)) + 1;
                        Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim * n);
                        Vector<double> testX = Vector<double>.Build.Random(i);

                        Network network = new Network();
                        network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: n, stride: 1));
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
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_WithStride1_MultipleFilters()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 8; i++)
                squares.Add((int)Math.Pow(i, 2));

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    for(int n = 1; n < 16; n++)
                    {
                        int outdim = (int)Math.Floor(Math.Sqrt(i) - Math.Sqrt(j)) + 1;
                        Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim * n);
                        Vector<double> testX = Vector<double>.Build.Random(i);
                        Vector<double> testWeights = Vector<double>.Build.Random(j * n);

                        Network network = new Network();
                        network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: n, stride: 1));
                        network.Add(new ActivationLayer(ActivationType.Tanh));
                        network.Add(new SoftmaxActivationLayer());
                        network.UseLoss(LossType.CategoricalCrossentropy);

                        double networkLossWithWeightAsVariable(Vector<double> x)
                        {
                            ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                            for (int k = 0; k < n; k++)
                            {
                                Vector<double> flattenedWeight = Vector<double>.Build.Dense(j);
                                for (int y = 0; y < j; y++)
                                    flattenedWeight[y] = x[y + j * k];
                                conv.Weights[k] = MathUtils.Unflatten(flattenedWeight);
                            }
                            Vector<double> output = network.Predict(testX);
                            return network.Loss(truthY, output);
                        }

                        Vector<double> finiteDiffWeightGradient = MathUtils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeights);
                        List<double> explicitWeightGradientList = new List<double>();

                        Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                        for (int k = network.Layers.Count - 1; k >= 0; k--)
                        {
                            outputGradient = network.Layers[k].BackPropagation(outputGradient);
                            if (k == 0) // retrieve weight gradient from convolutional layer
                            {
                                ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                                for (int y = 0; y < conv.WeightGradients.Length; y++)
                                {
                                    Vector<double> weightGrad = MathUtils.Flatten(conv.WeightGradients[y]);
                                    for (int q = 0; q < weightGrad.Count; q++)
                                        explicitWeightGradientList.Add(weightGrad[q]);
                                }
                            }
                        }

                        Vector<double> explicitWeightGradient = Vector<double>.Build.DenseOfEnumerable(explicitWeightGradientList);

                        Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.0001);
                    }
                }
            }


        }

        [Test]
        public void ConvolutionLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_WithStrideMoreThan1_MultipleFilters()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 16; i+=2)
                squares.Add((int)Math.Pow(i, 2));
            List<int> nums = new List<int>();
            for (int i = 1; i < 16; i++)
                nums.Add(i);

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    for (int n = 1; n < 8; n++) // filter count
                    {
                        foreach(int s in nums.Where(x => Math.Abs(Math.Floor((Math.Sqrt(i) - Math.Sqrt(j)) / x) - ((Math.Sqrt(i) - Math.Sqrt(j)) / x)) < 0.0000001)) //multiple stride where stride goes into the image evenly
                        {
                            int outdim = (int)Math.Floor((Math.Sqrt(i) - Math.Sqrt(j))/s) + 1;
                            Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim * n);
                            Vector<double> testX = Vector<double>.Build.Random(i);

                            Network network = new Network();
                            network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: n, stride: s));
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
        }

        [Test]
        public void ConvolutionLayer_ReturnsCorrectWeightGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxTanhAndConvolutionalLayer_WithStrideMoreThan1_MultipleFilters()
        {
            List<int> squares = new List<int>();
            for (int i = 1; i < 16; i+=2)
                squares.Add((int)Math.Pow(i, 2));
            List<int> nums = new List<int>();
            for (int i = 1; i < 16; i++)
                nums.Add(i);

            foreach (int i in squares)
            {
                foreach (int j in squares.Where(s => s <= i)) //test every square kernel up to the size of the input matrix
                {
                    for (int n = 1; n < 8; n++)
                    {
                        foreach (int s in nums.Where(x => Math.Abs(Math.Floor((Math.Sqrt(i) - Math.Sqrt(j)) / x) - ((Math.Sqrt(i) - Math.Sqrt(j)) / x)) < 0.0000001)) //multiple stride where stride goes into the image evenly
                        {
                            int outdim = (int)Math.Floor((Math.Sqrt(i) - Math.Sqrt(j))/s) + 1;
                            Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim * n);
                            Vector<double> testX = Vector<double>.Build.Random(i);
                            Vector<double> testWeights = Vector<double>.Build.Random(j * n);

                            Network network = new Network();
                            network.Add(new ConvolutionalLayer(i, kernel: (int)Math.Sqrt(j), filters: n, stride: s));
                            network.Add(new ActivationLayer(ActivationType.Tanh));
                            network.Add(new SoftmaxActivationLayer());
                            network.UseLoss(LossType.CategoricalCrossentropy);

                            double networkLossWithWeightAsVariable(Vector<double> x)
                            {
                                ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                                for (int k = 0; k < n; k++)
                                {
                                    Vector<double> flattenedWeight = Vector<double>.Build.Dense(j);
                                    for (int y = 0; y < j; y++)
                                        flattenedWeight[y] = x[y + j * k];
                                    conv.Weights[k] = MathUtils.Unflatten(flattenedWeight);
                                }
                                Vector<double> output = network.Predict(testX);
                                return network.Loss(truthY, output);
                            }

                            Vector<double> finiteDiffWeightGradient = MathUtils.FiniteDifferencesGradient(networkLossWithWeightAsVariable, testWeights);
                            List<double> explicitWeightGradientList = new List<double>();

                            Vector<double> outputGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));

                            for (int k = network.Layers.Count - 1; k >= 0; k--)
                            {
                                outputGradient = network.Layers[k].BackPropagation(outputGradient);
                                if (k == 0) // retrieve weight gradient from convolutional layer
                                {
                                    ConvolutionalLayer conv = (ConvolutionalLayer)network.Layers[0];
                                    for (int y = 0; y < conv.WeightGradients.Length; y++)
                                    {
                                        Vector<double> weightGrad = MathUtils.Flatten(conv.WeightGradients[y]);
                                        for (int q = 0; q < weightGrad.Count; q++)
                                            explicitWeightGradientList.Add(weightGrad[q]);
                                    }
                                }
                            }

                            Vector<double> explicitWeightGradient = Vector<double>.Build.DenseOfEnumerable(explicitWeightGradientList);

                            Assert.IsTrue((finiteDiffWeightGradient - explicitWeightGradient).L2Norm() < 0.0001);
                        }
                    }
                }
            }


        }
        #endregion

        #region Operator Tests
        [Test]
        public void Convolution_ReturnsCorrectConvolution_WhenPassedMatrixAndFilterWithStride()
        {
            #region Convolution Setup 1
            double[] mtx = new double[]
            {
                  1, 2, 3,
                  4, 5, 6,
                  7, 8, 9
            };

            double[,] filt = new double[,]
            {
                { 4, 3 },
                { 2, 1 }
            };

            double[,] exp = new double[,]
            {
                { 23, 33 },
                { 53, 63 }
            };

            Vector<double> mtx1 = Vector<double>.Build.DenseOfArray(mtx);
            Matrix<double> filt1 = Matrix<double>.Build.DenseOfArray(filt);
            Matrix<double> expected1 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 2
            mtx = new double[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                  13, 2, 15, 7
            };

            filt = new double[,]
            {
                { 5, 2 },
                { 8, 7 }
            };

            exp = new double[,]
            {
                { 76, 98, 160 },
                { 142, 164, 208 },
                { 169, 179, 238 }
            };

            Vector<double> mtx2 = Vector<double>.Build.DenseOfArray(mtx);
            Matrix<double> filt2 = Matrix<double>.Build.DenseOfArray(filt);
            Matrix<double> expected2 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 3
            mtx = new double[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                  13, 2, 15, 7
            };

            filt = new double[,]
            {
                { 5, 2 },
                { 8, 7 }
            };

            exp = new double[,]
            {
                { 76, 160 },
                { 169, 238 }
            };

            Vector<double> mtx3 = Vector<double>.Build.DenseOfArray(mtx);
            Matrix<double> filt3 = Matrix<double>.Build.DenseOfArray(filt);
            Matrix<double> expected3 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 4
            mtx = new double[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new double[,]
            {
                { 5, 2, 3 },
                { 8, 7, 4 },
                { 9, 8, 5 }
            };

            exp = new double[,]
            {
                { 281, 376 },
                { 404, 418 }
            };

            Vector<double> mtx4 = Vector<double>.Build.DenseOfArray(mtx);
            Matrix<double> filt4 = Matrix<double>.Build.DenseOfArray(filt);
            Matrix<double> expected4 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 5
            mtx = new double[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new double[,]
            {
                { 5, 2, 3 },
                { 8, 7, 4 },
                { 9, 8, 5 }
            };

            exp = new double[,]
            {
                { 281 }
            };

            Vector<double> mtx5 = Vector<double>.Build.DenseOfArray(mtx);
            Matrix<double> filt5 = Matrix<double>.Build.DenseOfArray(filt);
            Matrix<double> expected5 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 6
            mtx = new double[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new double[,]
            {
                { 5 }
            };

            exp = new double[,]
            {
                 { 5, 10, 15, 50 },
                 { 20, 25, 30, 55 },
                 { 35, 40, 45, 60 },
                 { 65, 10, 75, 35 }
            };

            Vector<double> mtx6 = Vector<double>.Build.DenseOfArray(mtx);
            Matrix<double> filt6 = Matrix<double>.Build.DenseOfArray(filt);
            Matrix<double> expected6 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 7
            mtx = new double[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new double[,]
            {
                { 3 }
            };

            exp = new double[,]
            {
                 { 3, 30 },
                 { 39, 21 }
            };

            Vector<double> mtx7 = Vector<double>.Build.DenseOfArray(mtx);
            Matrix<double> filt7 = Matrix<double>.Build.DenseOfArray(filt);
            Matrix<double> expected7 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 8
            mtx = new double[]
            {
                  5, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new double[,]
            {
                { 3 }
            };

            exp = new double[,]
            {
                 { 15 }
            };

            Vector<double> mtx8 = Vector<double>.Build.DenseOfArray(mtx);
            Matrix<double> filt8 = Matrix<double>.Build.DenseOfArray(filt);
            Matrix<double> expected8 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 9
            mtx = new double[]
            {
                  5, 2, 3, 10, 6,
                  4, 5, 6, 11, 7,
                  7, 8, 9, 12, 2,
                 13, 2, 15, 7, 1,
                 10, 3, 11, 6, 5
            };

            filt = new double[,]
            {
                { 5, 2, 3 },
                { 8, 7, 4 },
                { 9, 8, 5 }
            };

            exp = new double[,]
            {
                 { 301, 393 },
                 { 425, 420 }
            };

            Vector<double> mtx9 = Vector<double>.Build.DenseOfArray(mtx);
            Matrix<double> filt9 = Matrix<double>.Build.DenseOfArray(filt);
            Matrix<double> expected9 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 10
            mtx = new double[]
            {
                  5, 2, 3, 10, 6,
                  4, 5, 6, 11, 7,
                  7, 8, 9, 12, 2,
                 13, 2, 15, 7, 1,
                 10, 3, 11, 6, 5
            };

            filt = new double[,]
            {
                { 5, 2 },
                { 8, 7 }
            };

            exp = new double[,]
            {
                 { 96, 199 },
                 { 170, 120 }
            };

            Vector<double> mtx10 = Vector<double>.Build.DenseOfArray(mtx);
            Matrix<double> filt10 = Matrix<double>.Build.DenseOfArray(filt);
            Matrix<double> expected10 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion

            Assert.AreEqual(expected1, ConvolutionalLayer.Convolution(mtx1, filt1, 1).Item2);
            Assert.AreEqual(expected2, ConvolutionalLayer.Convolution(mtx2, filt2, 1).Item2);
            Assert.AreEqual(expected3, ConvolutionalLayer.Convolution(mtx3, filt3, 2).Item2);
            Assert.AreEqual(expected5, ConvolutionalLayer.Convolution(mtx5, filt5, 2).Item2);
            Assert.AreEqual(expected6, ConvolutionalLayer.Convolution(mtx6, filt6, 1).Item2);
            Assert.AreEqual(expected7, ConvolutionalLayer.Convolution(mtx7, filt7, 3).Item2);
            Assert.AreEqual(expected8, ConvolutionalLayer.Convolution(mtx8, filt8, 4).Item2);
            Assert.AreEqual(expected9, ConvolutionalLayer.Convolution(mtx9, filt9, 2).Item2);
            Assert.AreEqual(expected10, ConvolutionalLayer.Convolution(mtx10, filt10, 3).Item2);
        }

        [Test]
        public void Dilate_ReturnsDilatedFilterGradient_WhenPassedPreviousLayerGradient()
        {
            #region Dilation Setup 1
            double[,] grad = new double[,]
            {
                { 4, 3 },
                { 2, 1 }
            };

            double[,] exp = new double[,]
            {
                { 4, 0, 3 },
                { 0, 0, 0 },
                { 2, 0, 1 },
            };

            Matrix<double> grad1 = Matrix<double>.Build.DenseOfArray(grad);
            Matrix<double> expected1 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Dilation Setup 2
            grad = new double[,]
            {
                { 4, 3, 5 },
                { 2, 1, 6 },
                { 5, 6, 7 },
            };

            exp = new double[,]
            {
                { 4, 0, 3, 0, 5},
                { 0, 0, 0, 0, 0},
                { 2, 0, 1, 0, 6},
                { 0, 0, 0, 0, 0},
                { 5, 0, 6, 0, 7}
            };

            Matrix<double> grad2 = Matrix<double>.Build.DenseOfArray(grad);
            Matrix<double> expected2 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Dilation Setup 3
            grad = new double[,]
            {
                { 4, 3, 5 },
                { 2, 1, 6 },
                { 5, 6, 7 },
            };

            exp = new double[,]
            {
                { 4, 0, 0, 3, 0, 0, 5 },
                { 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0 },
                { 2, 0, 0, 1, 0, 0, 6 },
                { 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0 },
                { 5, 0, 0, 6, 0, 0, 7 },
            };

            Matrix<double> grad3 = Matrix<double>.Build.DenseOfArray(grad);
            Matrix<double> expected3 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Dilation Setup 4
            grad = new double[,]
            {
                { 4, 3, 5 },
                { 2, 1, 6 },
                { 5, 6, 7 },
            };

            exp = new double[,]
            {
                { 4, 3, 5 },
                { 2, 1, 6 },
                { 5, 6, 7 },
            };

            Matrix<double> grad4 = Matrix<double>.Build.DenseOfArray(grad);
            Matrix<double> expected4 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion

            Assert.AreEqual(expected1, ConvolutionalLayer.Dilate(grad1, 2));
            Assert.AreEqual(expected2, ConvolutionalLayer.Dilate(grad2, 2));
            Assert.AreEqual(expected3, ConvolutionalLayer.Dilate(grad3, 3));
            Assert.AreEqual(expected4, ConvolutionalLayer.Dilate(grad4, 1));
        }

        [Test]
        public void PadAndDilate_ReturnsPaddedAndDilatedMatrix_WhenPassedGradientMatrixStrideAndKernel()
        {
            #region Pan and Dilate Setup 1
            double[,] grad = new double[,]
            {
                { 4, 3 },
                { 2, 1 }
            };

            double[,] exp = new double[,]
            {
                { 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 4, 0, 3, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 2, 0, 1, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0 },
            };

            Matrix<double> grad1 = Matrix<double>.Build.DenseOfArray(grad);
            Matrix<double> expected1 = Matrix<double>.Build.DenseOfArray(exp).Transpose();
            #endregion
            #region Pan and Dilate Setup 2
            grad = new double[,]
            {
                { 4, 3, 5 },
                { 2, 1, 9 },
                { 6, 8, 2 }
            };

            exp = new double[,]
            {
                { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 4, 0, 3, 0, 5, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 2, 0, 1, 0, 9, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 6, 0, 8, 0, 2, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            };

            Matrix<double> grad2 = Matrix<double>.Build.DenseOfArray(grad);
            Matrix<double> expected2 = Matrix<double>.Build.DenseOfArray(exp).Transpose();
            #endregion
            #region Pan and Dilate Setup 3
            grad = new double[,]
            {
                { 4, 3, 5 },
                { 2, 1, 9 },
                { 6, 8, 2 }
            };

            exp = new double[,]
            {
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 4, 0, 0, 3, 0, 0, 5, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 2, 0, 0, 1, 0, 0, 9, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 6, 0, 0, 8, 0, 0, 2, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            };

            Matrix<double> grad3 = Matrix<double>.Build.DenseOfArray(grad);
            Matrix<double> expected3 = Matrix<double>.Build.DenseOfArray(exp).Transpose();
            #endregion

            Assert.AreEqual(expected1, ConvolutionalLayer.PadAndDilate(grad1, 2, 3));
            Assert.AreEqual(expected2, ConvolutionalLayer.PadAndDilate(grad2, 2, 3));
            Assert.AreEqual(expected3, ConvolutionalLayer.PadAndDilate(grad3, 3, 3));
        }

        [Test]
        public void Rotate180_ReturnsMatrixFlipped180Degrees_WhenPassedMatrix()
        {
            #region Rotate 180 Setup 1
            double[,] mtx = new double[,]
            {
                { 4, 3 },
                { 2, 1 }
            };

            double[,] exp = new double[,]
            {
                { 1, 2 },
                { 3, 4 }
            };

            Matrix<double> mtx1 = Matrix<double>.Build.DenseOfArray(mtx);
            Matrix<double> expected1 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Rotate 180 Setup 2
            mtx = new double[,]
            {
                { 4, 3, 8 },
                { 2, 1, 7 },
                { 7, 2, 5 },
            };

            exp = new double[,]
            {
                { 5, 2, 7 },
                { 7, 1, 2 },
                { 8, 3, 4 },
            };

            Matrix<double> mtx2 = Matrix<double>.Build.DenseOfArray(mtx);
            Matrix<double> expected2 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Rotate 180 Setup 3
            mtx = new double[,]
            {
                { 4, 3, 8, 1 },
                { 2, 1, 7, 3 },
                { 7, 2, 5, 9 },
                { 4, 6, 4, 7 }
            };
            exp = new double[,]
            {
                { 7, 4, 6, 4 },
                { 9, 5, 2, 7 },
                { 3, 7, 1, 2 },             
                { 1, 8, 3, 4 },
            };

            Matrix<double> mtx3 = Matrix<double>.Build.DenseOfArray(mtx);
            Matrix<double> expected3 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion
            #region Rotate 180 Setup 4
            mtx = new double[,]
            {
                { 4 }
            };
            exp = new double[,]
            {
                { 4 }
            };

            Matrix<double> mtx4 = Matrix<double>.Build.DenseOfArray(mtx);
            Matrix<double> expected4 = Matrix<double>.Build.DenseOfArray(exp);
            #endregion

            Assert.AreEqual(expected1, ConvolutionalLayer.Rotate180(mtx1));
            Assert.AreEqual(expected2, ConvolutionalLayer.Rotate180(mtx2));
            Assert.AreEqual(expected3, ConvolutionalLayer.Rotate180(mtx3));
            Assert.AreEqual(expected4, ConvolutionalLayer.Rotate180(mtx4));
        }
        #endregion
    }
}