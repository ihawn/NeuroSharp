using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;

namespace UnitTests
{
    public class MaxPoolTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void MaxPool_ReturnsCorrectMaxPool_WhenPassedSquareMatrix()
        {
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
            List<(int, int)> expectedPositions1 = new List<(int, int)>() { (1, 1), (1, 2), (2, 0), (1, 2), };

            Assert.AreEqual(expected1, MaxPoolingLayer.MaxPool(matrix1, 2, 1).Item1);
            Assert.AreEqual(expectedPositions1, MaxPoolingLayer.MaxPool(matrix1, 2, 1).Item2);
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