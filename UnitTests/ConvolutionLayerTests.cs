using NUnit.Framework;
using NeuroSharp;
using MathNet.Numerics.LinearAlgebra;

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
            float[] mtx = new float[]
            {
                  1, 2, 3,
                  4, 5, 6,
                  7, 8, 9
            };

            float[,] filt = new float[,]
            {
                { 4, 3 },
                { 2, 1 }
            };

            float[,] exp = new float[,]
            {
                { 23, 33 },
                { 53, 63 }
            };

            Vector<float> mtx1 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt1 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected1 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 2
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                  13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 5, 2 },
                { 8, 7 }
            };

            exp = new float[,]
            {
                { 76, 98, 160 },
                { 142, 164, 208 },
                { 169, 179, 238 }
            };

            Vector<float> mtx2 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt2 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected2 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 3
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                  13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 5, 2 },
                { 8, 7 }
            };

            exp = new float[,]
            {
                { 76, 160 },
                { 169, 238 }
            };

            Vector<float> mtx3 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt3 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected3 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 4
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 5, 2, 3 },
                { 8, 7, 4 },
                { 9, 8, 5 }
            };

            exp = new float[,]
            {
                { 281, 376 },
                { 404, 418 }
            };

            Vector<float> mtx4 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt4 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected4 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 5
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 5, 2, 3 },
                { 8, 7, 4 },
                { 9, 8, 5 }
            };

            exp = new float[,]
            {
                { 281 }
            };

            Vector<float> mtx5 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt5 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected5 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 6
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 5 }
            };

            exp = new float[,]
            {
                 { 5, 10, 15, 50 },
                 { 20, 25, 30, 55 },
                 { 35, 40, 45, 60 },
                 { 65, 10, 75, 35 }
            };

            Vector<float> mtx6 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt6 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected6 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 7
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 3 }
            };

            exp = new float[,]
            {
                 { 3, 30 },
                 { 39, 21 }
            };

            Vector<float> mtx7 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt7 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected7 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 8
            mtx = new float[]
            {
                  5, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 3 }
            };

            exp = new float[,]
            {
                 { 15 }
            };

            Vector<float> mtx8 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt8 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected8 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 9
            mtx = new float[]
            {
                  5, 2, 3, 10, 6,
                  4, 5, 6, 11, 7,
                  7, 8, 9, 12, 2,
                 13, 2, 15, 7, 1,
                 10, 3, 11, 6, 5
            };

            filt = new float[,]
            {
                { 5, 2, 3 },
                { 8, 7, 4 },
                { 9, 8, 5 }
            };

            exp = new float[,]
            {
                 { 301, 393 },
                 { 425, 420 }
            };

            Vector<float> mtx9 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt9 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected9 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Convolution Setup 10
            mtx = new float[]
            {
                  5, 2, 3, 10, 6,
                  4, 5, 6, 11, 7,
                  7, 8, 9, 12, 2,
                 13, 2, 15, 7, 1,
                 10, 3, 11, 6, 5
            };

            filt = new float[,]
            {
                { 5, 2 },
                { 8, 7 }
            };

            exp = new float[,]
            {
                 { 96, 199 },
                 { 170, 120 }
            };

            Vector<float> mtx10 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt10 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected10 = Matrix<float>.Build.DenseOfArray(exp);
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
        public void PadAndDilate_ReturnsPaddedAndDilatedMatrix_WhenPassedGradientMatrixStrideAndKernel()
        {
            #region Pan and Dilate Setup 1
            float[,] grad = new float[,]
            {
                { 4, 3 },
                { 2, 1 }
            };

            float[,] exp = new float[,]
            {
                { 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 4, 0, 3, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 2, 0, 1, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0 },
                { 0, 0, 0, 0, 0, 0, 0 },
            };

            Matrix<float> grad1 = Matrix<float>.Build.DenseOfArray(grad);
            Matrix<float> expected1 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Pan and Dilate Setup 2
            grad = new float[,]
            {
                { 4, 3, 5 },
                { 2, 1, 9 },
                { 6, 8, 2 }
            };

            exp = new float[,]
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

            Matrix<float> grad2 = Matrix<float>.Build.DenseOfArray(grad);
            Matrix<float> expected2 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Pan and Dilate Setup 3
            grad = new float[,]
            {
                { 4, 3, 5 },
                { 2, 1, 9 },
                { 6, 8, 2 }
            };

            exp = new float[,]
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

            Matrix<float> grad3 = Matrix<float>.Build.DenseOfArray(grad);
            Matrix<float> expected3 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion

            Assert.AreEqual(expected1, ConvolutionalLayer.PadAndDilate(grad1, 2, 3));
            Assert.AreEqual(expected2, ConvolutionalLayer.PadAndDilate(grad2, 2, 3));
            Assert.AreEqual(expected3, ConvolutionalLayer.PadAndDilate(grad3, 3, 3));
        }

        /*[Test]
        public void BackwardsConvolution_ReturnsCorrectBackwardsConvolution_WhenPassedMatrixAndFilterWithStride()
        {
            #region Backwards Convolution Setup 1
            float[] mtx = new float[]
            {
                  1, 2, 3,
                  4, 5, 6,
                  7, 8, 9
            };

            float[,] filt = new float[,]
            {
                { 4, 3 },
                { 2, 1 }
            };

            float[,] exp = new float[,]
            {
                { 23, 33 },
                { 53, 63 }
            };

            Vector<float> mtx1 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt1 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected1 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Backwards Convolution Setup 2
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                  13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 5, 2 },
                { 8, 7 }
            };

            exp = new float[,]
            {
                { 76, 98, 160 },
                { 142, 164, 208 },
                { 169, 179, 238 }
            };

            Vector<float> mtx2 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt2 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected2 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Backwards Convolution Setup 3
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                  13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 5, 2 },
                { 8, 7 }
            };

            exp = new float[,]
            {
                { 76, 160 },
                { 169, 238 }
            };

            Vector<float> mtx3 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt3 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected3 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Backwards Convolution Setup 4
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 5, 2, 3 },
                { 8, 7, 4 },
                { 9, 8, 5 }
            };

            exp = new float[,]
            {
                { 281, 376 },
                { 404, 418 }
            };

            Vector<float> mtx4 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt4 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected4 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Backwards Convolution Setup 5
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 5, 2, 3 },
                { 8, 7, 4 },
                { 9, 8, 5 }
            };

            exp = new float[,]
            {
                { 281 }
            };

            Vector<float> mtx5 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt5 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected5 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Backwards Convolution Setup 6
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 5 }
            };

            exp = new float[,]
            {
                 { 5, 10, 15, 50 },
                 { 20, 25, 30, 55 },
                 { 35, 40, 45, 60 },
                 { 65, 10, 75, 35 }
            };

            Vector<float> mtx6 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt6 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected6 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Backwards Convolution Setup 7
            mtx = new float[]
            {
                  1, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 3 }
            };

            exp = new float[,]
            {
                 { 3, 30 },
                 { 39, 21 }
            };

            Vector<float> mtx7 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt7 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected7 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Backwards Convolution Setup 8
            mtx = new float[]
            {
                  5, 2, 3, 10,
                  4, 5, 6, 11,
                  7, 8, 9, 12,
                 13, 2, 15, 7
            };

            filt = new float[,]
            {
                { 3 }
            };

            exp = new float[,]
            {
                 { 15 }
            };

            Vector<float> mtx8 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt8 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected8 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Backwards Convolution Setup 9
            mtx = new float[]
            {
                  5, 2, 3, 10, 6,
                  4, 5, 6, 11, 7,
                  7, 8, 9, 12, 2,
                 13, 2, 15, 7, 1,
                 10, 3, 11, 6, 5
            };

            filt = new float[,]
            {
                { 5, 2, 3 },
                { 8, 7, 4 },
                { 9, 8, 5 }
            };

            exp = new float[,]
            {
                 { 301, 393 },
                 { 425, 420 }
            };

            Vector<float> mtx9 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt9 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected9 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion
            #region Backwards Convolution Setup 10
            mtx = new float[]
            {
                  5, 2, 3, 10, 6,
                  4, 5, 6, 11, 7,
                  7, 8, 9, 12, 2,
                 13, 2, 15, 7, 1,
                 10, 3, 11, 6, 5
            };

            filt = new float[,]
            {
                { 5, 2 },
                { 8, 7 }
            };

            exp = new float[,]
            {
                 { 96, 199 },
                 { 170, 120 }
            };

            Vector<float> mtx10 = Vector<float>.Build.DenseOfArray(mtx);
            Matrix<float> filt10 = Matrix<float>.Build.DenseOfArray(filt);
            Matrix<float> expected10 = Matrix<float>.Build.DenseOfArray(exp);
            #endregion

            Assert.AreEqual(expected1, ConvolutionalLayer.BackwardsConvolution(mtx1, filt1, 1));
        }*/
    }
}