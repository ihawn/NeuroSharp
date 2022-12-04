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
        public void MaxPoolLayer_ReturnsCorrectInputGradient_WhenChainedTogetherWithCategoricalCrossentropySoftmaxReluConvolution_MultipleFilterInput_StrideMoreThan1()
        {
            for(int i = 1; i <= 5; i++)
            {
                for (int j = 1; j <= 5; j++)
                {
                    Vector<double> truthY = Vector<double>.Build.Random(5);
                    Vector<double> testX = Vector<double>.Build.Random(64 * 3);

                    Network network = new Network(64 * 3);
                    network.Add(new ConvolutionalLayer(kernel: 2, filters: 5, stride: 1, channels: 3));
                    network.Add(new ActivationLayer(ActivationType.ReLu));
                    network.Add(new MaxPoolingLayer(poolSize: i, stride: j));
                    network.Add(new FullyConnectedLayer(5));
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
            List<XYPair> expectedPositions1 = new List<XYPair>() { new (1, 1), new (1, 2), new (2, 0), new (1, 2) };
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
            List<XYPair> expectedPositions2 = new List<XYPair>() { new (0, 0), new (1, 2), new (2, 0), new (2, 2) };
            #endregion

            Assert.AreEqual(expected1, MaxPoolingLayer.MaxPool(matrix1, 2, 1).Item1);
            Assert.AreEqual(expectedPositions1, MaxPoolingLayer.MaxPool(matrix1, 2, 1).Item2);

            Assert.AreEqual(expected2, MaxPoolingLayer.MaxPool(matrix2, 2, 1).Item1);
            Assert.AreEqual(expectedPositions2, MaxPoolingLayer.MaxPool(matrix2, 2, 1).Item2);
        }

        [Test]
        public void MaxPool_ReturnsCorrectMaxPool_WhenPassedSquareMatrixStrideMoreThan1()
        {
            #region Max Pool Setup 1
            double[,] mtx = new double[,]
            {
                { 1, 2, 3, 4 },
                { 5, 6, 7, 8 },
                { 9, 10, 11, 12 },
                { 13, 14, 15, 16 },
            };

            double[,] exp = new double[,]
            {
                { 6, 8 },
                { 14, 16 }
            };

            Matrix<double> matrix1 = Matrix<double>.Build.DenseOfArray(mtx);
            Matrix<double> expected1 = Matrix<double>.Build.DenseOfArray(exp);
            List<XYPair> expectedPositions1 = new List<XYPair>() { new (1, 1), new (1, 3), new (3, 1), new (3, 3) };
            #endregion
            #region Max Pool Setup 2
            mtx = new double[,]
            {
                { 1, 2, 3, 4, 17, 18 },
                { 5, 6, 7, 8, 19, 20 },
                { 9, 10, 11, 12, 21, 22 },
                { 13, 14, 15, 16, 23, 24 },
                { 25, 26, 27, 28, 36, 30 },
                { 31, 32, 33, 34, 35, 29 },
            };

            exp = new double[,]
            {
                { 11, 22 },
                { 33, 36 }
            };

            Matrix<double> matrix2 = Matrix<double>.Build.DenseOfArray(mtx);
            Matrix<double> expected2 = Matrix<double>.Build.DenseOfArray(exp);
            List<XYPair> expectedPositions2 = new List<XYPair>() { new (2, 2), new (2, 5), new (5, 2), new (4, 4) };
            #endregion
            
            #region Max Pool Setup 3
            mtx = new double[,]
            {
                { 1, 2, 3, 4, 17, 18 },
                { 5, 6, 7, 8, 19, 20 },
                { 9, 10, 11, 12, 21, 22 },
                { 13, 14, 15, 16, 23, 24 },
                { 25, 26, 27, 28, 36, 30 },
                { 31, 32, 33, 34, 35, 29 },
            };

            exp = new double[,]
            {
                { 16, 24 },
                { 34, 36 }
            };

            Matrix<double> matrix3 = Matrix<double>.Build.DenseOfArray(mtx);
            Matrix<double> expected3 = Matrix<double>.Build.DenseOfArray(exp);
            List<XYPair> expectedPositions3 = new List<XYPair>() { new (3, 3), new (3, 5), new (5, 3), new (4, 4) };
            #endregion

            Assert.AreEqual(expected1, MaxPoolingLayer.MaxPool(matrix1, 2, 2).Item1);
            Assert.AreEqual(expectedPositions1, MaxPoolingLayer.MaxPool(matrix1, 2, 2).Item2);

            Assert.AreEqual(expected2, MaxPoolingLayer.MaxPool(matrix2, 3, 3).Item1);
            Assert.AreEqual(expectedPositions2, MaxPoolingLayer.MaxPool(matrix2, 3, 3).Item2);
            
            Assert.AreEqual(expected3, MaxPoolingLayer.MaxPool(matrix2, 4, 2).Item1);
            Assert.AreEqual(expectedPositions3, MaxPoolingLayer.MaxPool(matrix2, 4, 2).Item2);
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
    }
}