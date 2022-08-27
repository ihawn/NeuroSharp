using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Utilities;
using NeuroSharp.Enumerations;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System;
using System.Linq;
using NeuroSharp.Training;

namespace UnitTests
{
    public class MaxPoolTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void MaxPoolLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmax()
        {
            List<int> squares = new List<int>();
            for (int u = 1; u < 10; u++)
                squares.Add((int)Math.Pow(u, 2));

            foreach (int u in squares)
            {
                foreach (int v in squares.Where(s => s <= u)) //test every square kernel up to the size of the input matrix
                {
                    int outdim = (int)Math.Floor(Math.Sqrt(u) - Math.Sqrt(v)) + 1;
                    Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim);
                    Vector<double> testX = Vector<double>.Build.Random(u);

                    Network network = new Network(u);
                    network.Add(new MaxPoolingLayer(poolSize: (int)Math.Sqrt(v)));
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
        public void MaxPoolLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmax_MultipleFilterInput()
        {
            List<int> squares = new List<int>();
            for (int u = 1; u < 10; u++)
                squares.Add((int)Math.Pow(u, 2));

            foreach (int u in squares)
            {
                foreach (int v in squares.Where(s => s <= u)) //test every square kernel up to the size of the input matrix
                {
                    for(int q = 1; q < 8; q++) //filter count
                    {
                        int outdim = (int)Math.Floor(Math.Sqrt(u) - Math.Sqrt(v)) + 1;
                        Vector<double> truthY = Vector<double>.Build.Random(outdim * outdim * q);
                        Vector<double> testX = Vector<double>.Build.Random(u * q);

                        Network network = new Network(u * q);
                        network.Add(new MaxPoolingLayer(poolSize: (int)Math.Sqrt(v)));
                        ((MaxPoolingLayer)network.Layers[0]).SetFilterCount(q);
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
        public void MaxPoolLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxReluConvolution_MultipleFilterInput()
        {
            for(int i = 0; i < 10; i++)
            {
                Vector<double> truthY = Vector<double>.Build.Random(4 * 5);
                Vector<double> testX = Vector<double>.Build.Random(16);

                Network network = new Network(16);
                network.Add(new ConvolutionalLayer(kernel: 2, filters: 5, stride: 1));
                network.Add(new ActivationLayer(ActivationType.ReLu));
                network.Add(new MaxPoolingLayer(poolSize: 2));
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

            for (int i = 0; i < 10; i++)
            {
                Vector<double> truthY = Vector<double>.Build.Random(4 * 4 * 20);
                Vector<double> testX = Vector<double>.Build.Random(100);

                Network network = new Network(100);
                network.Add(new ConvolutionalLayer(kernel: 5, filters: 20, stride: 1));
                network.Add(new ActivationLayer(ActivationType.ReLu));
                network.Add(new MaxPoolingLayer(poolSize: 3));
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
        public void MaxPool_ReturnsCorrectMaxPool_WhenPassedSquareMatrix()
        {
            #region Max Pool Setup 1
            double[,] mtx = new double[,]
            {
                { 1, 2, 3 },
                { 5, 6, 7 },
                { 9, 5, 1 }
            };

            double[,] exp = new double[,]
            {
                { 6, 7 },
                { 9, 7 }
            };

            Matrix<double> matrix1 = Matrix<double>.Build.DenseOfArray(mtx);
            Matrix<double> expected1 = Matrix<double>.Build.DenseOfArray(exp);
            List<(int, int)> expectedPositions1 = new List<(int, int)>() { (1, 1), (1, 2), (2, 0), (1, 2) };
            #endregion
            #region Max Pool Setup 2
            mtx = new double[,]
            {
                { 10, 2, 3 },
                { 5, 6, 7 },
                { 9, 5, 11 }
            };

            exp = new double[,]
            {
                { 10, 7 },
                { 9, 11 }
            };

            Matrix<double> matrix2 = Matrix<double>.Build.DenseOfArray(mtx);
            Matrix<double> expected2 = Matrix<double>.Build.DenseOfArray(exp);
            List<(int, int)> expectedPositions2 = new List<(int, int)>() { (0, 0), (1, 2), (2, 0), (2, 2) };
            #endregion

            Assert.AreEqual(expected1, MaxPoolingLayer.MaxPool(matrix1, 2, 1).Item1);
            Assert.AreEqual(expectedPositions1, MaxPoolingLayer.MaxPool(matrix1, 2, 1).Item2);

            Assert.AreEqual(expected2, MaxPoolingLayer.MaxPool(matrix2, 2, 1).Item1);
            Assert.AreEqual(expectedPositions2, MaxPoolingLayer.MaxPool(matrix2, 2, 1).Item2);
        }

        [Test]
        public void SliceFlattenedMatrixIntoSquares_ReturnsListOfSquareMatrices_WhenPassesFlattenedListOfConcatMatrices()
        {
            #region Slicer Setup 1
            double[] lst = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 };
            double[,] e1 = new double[,] { { 1, 2, 3, }, { 4, 5, 6 }, { 7, 8, 9 } };
            double[,] e2 = new double[,] { { 10, 11, 12, }, { 13, 14, 15 }, { 16, 17, 18 } };
            double[,] e3 = new double[,] { { 19, 20, 21, }, { 22, 23, 24 }, { 25, 26, 27 } };
            Matrix<double> m1 = Matrix<double>.Build.DenseOfArray(e1);
            Matrix<double> m2 = Matrix<double>.Build.DenseOfArray(e2);
            Matrix<double> m3 = Matrix<double>.Build.DenseOfArray(e3);
            List<Matrix<double>> expected1 = new List<Matrix<double>>() { m1, m2, m3 };
            Vector<double> input1 = Vector<double>.Build.DenseOfArray(lst);
            #endregion
            #region Slicer Setup 2
            lst = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 };
            e1 = new double[,] { { 1, 2, 3, }, { 4, 5, 6 }, { 7, 8, 9 } };
            e2 = new double[,] { { 10, 11, 12, }, { 13, 14, 15 }, { 16, 17, 18 } };
            m1 = Matrix<double>.Build.DenseOfArray(e1);
            m2 = Matrix<double>.Build.DenseOfArray(e2);
            List<Matrix<double>> expected2 = new List<Matrix<double>>() { m1, m2 };
            Vector<double> input2 = Vector<double>.Build.DenseOfArray(lst);
            #endregion

            Assert.AreEqual(expected1, MaxPoolingLayer.SliceFlattenedMatrixIntoSquares(input1, 3));
            Assert.AreEqual(expected2, MaxPoolingLayer.SliceFlattenedMatrixIntoSquares(input2, 2));
        }
        
        //todo: write tests for max pooling with stride > 1
    }
}