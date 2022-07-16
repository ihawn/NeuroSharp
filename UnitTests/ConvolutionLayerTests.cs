using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System;

namespace UnitTests
{
    public class ConvolutionLayerTests
    {
        [SetUp]
        public void Setup()
        {
        }

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

        /*[Test]
        public void ComputeWeightGradient_ReturnsCorrectWeightGrad_WhenPassedPrevLayerInputAndJacobianAndStride()
        {
            #region Weight Gradient Setup 1
            ConvolutionalLayer c1 = new ConvolutionalLayer(4, kernel: 2, stride: 1);
            Matrix<double> randJacobian1 = Matrix<double>.Build.Random(1, 1);
            Vector<double> randInput1 = Vector<double>.Build.Random(4);
            Matrix<double> testGrad1 = ConvolutionalLayer.ComputeWeightGradient(c1.Input, randJacobian1, stride: 1);
            Vector<double> trueGrad1 = Utils.Flatten(randJacobian1 * Utils.FiniteDifferencesGradient(c1.ForwardPropagation, randInput1).ToRowMatrix()); // chain rule to mimic passed gradient from connected layer
            #endregion
        }*/

        [Test]
        public void ComputeInputGradient_ReturnsCorrectInputGrad_WhenPassedWeightGradStride()
        {
            #region Weight Gradient Setup 1
            ConvolutionalLayer c1 = new ConvolutionalLayer(4, kernel: 2, stride: 1);
            Matrix<double> randJacobian1 = Matrix<double>.Build.Random(1, 1);
            Vector<double> randInput1 = Vector<double>.Build.Random(4);
            Vector<double> testGrad1 = ConvolutionalLayer.ComputeInputGradient(c1.Weights, randJacobian1, stride: 1);
            Vector<double> trueGrad1 = Utils.Flatten(randJacobian1 * Utils.FiniteDifferencesGradient(c1.ForwardPropagation, randInput1).ToRowMatrix()); // chain rule to mimic passed gradient from connected layer
            #endregion
            /*#region Weight Gradient Setup 2
            ConvolutionalLayer c2 = new ConvolutionalLayer(9, kernel: 2, stride: 1);
            Matrix<double> randJacobian2 = Matrix<double>.Build.Random(2, 2);
            Vector<double> randInput2 = Vector<double>.Build.Random(9);
            Vector<double> testGrad2 = ConvolutionalLayer.ComputeInputGradient(c2.Weights, randJacobian2, stride: 1);
            Vector<double> trueGrad2 = Utils.FiniteDifferencesGradient(c2.ForwardPropagation, randInput2);
            #endregion*/

            Assert.IsTrue((trueGrad1 - testGrad1).L2Norm()/Math.Max(trueGrad1.L2Norm(), testGrad1.L2Norm()) < 0.00001d);
           // Assert.IsTrue((trueGrad2 - testGrad2).L2Norm() < 0.00001d);
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
            #region Dilation Setup 3
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
            Matrix<double> expected1 = Matrix<double>.Build.DenseOfArray(exp);
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
            Matrix<double> expected2 = Matrix<double>.Build.DenseOfArray(exp);
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
            Matrix<double> expected3 = Matrix<double>.Build.DenseOfArray(exp);
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
    }
}