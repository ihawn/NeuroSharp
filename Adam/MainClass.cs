using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;

namespace Adam
{
    public class MainClass
    {
        static void Main(string[] args)
        {
            //TestFromWeightBiasData();
            //TestFromFunction();

            Adam a = new Adam(Matrix<float>.Build.Random(1, 2));
            a.StepForX2Y2();
        }

        static void TestFromWeightBiasData()
        {
            float[,] weight1 = new float[,]
            {
                { 0.2227f, -0.0198f, -0.1030f,  0.0870f, -0.0629f, -0.0540f, -0.0798f,  0.1665f, -0.0068f,  0.0285f, -0.0188f },
                { 0.1218f,  0.1233f, -0.2650f, -0.0281f, -0.1503f,  0.1978f,  0.2136f,  0.0159f,  0.2188f, 0.2681f, -0.2405f },
                { 0.1548f, -0.2080f, -0.2429f,  0.2956f,  0.0951f, -0.1789f,  0.2437f,  0.0809f,  0.3009f,  0.0597f,  0.2489f },
                { -0.1085f, -0.2835f, -0.0254f, -0.0854f,  0.2432f, 0.0266f, -0.1151f, -0.2360f,  0.2140f, -0.0865f,  0.1179f }
            };
            float[] bias1 = new float[] { -0.1430f, -0.1882f, 0.0658f, -0.0196f };

            float[,] weight2 = new float[,]
            {
                { -0.3025f, -0.2036f, -0.3880f,  0.2955f },
                { -0.1374f,  0.1184f, -0.4437f,  0.4445f },
                { 0.1989f,  0.2389f,  0.3519f, -0.1063f }
            };
            float[] bias2 = new float[] { -0.1978f, 0.0826f, 0.3677f };

            float[,] weightGrad2 = new float[,]
            {
                { 0.0000f, 0.0000f, 0.0000f, 0.0000f },
                { 0.0000f, 0.0000f, 0.0000f, 0.0000f },
                { 0.0000f, 0.0000f, 0.0256f, 0.0091f }
            };
            float[] biasGrad2 = new float[] { 0.0000f, 0.0000f, 0.0086f };

            Matrix<float> weight = Matrix<float>.Build.DenseOfArray(weight2);
            Vector<float> bias = Vector<float>.Build.DenseOfArray(bias2);
            Matrix<float> weightGrad = Matrix<float>.Build.DenseOfArray(weightGrad2);
            Vector<float> biasGrad = Vector<float>.Build.DenseOfArray(biasGrad2);

            Adam adam = new Adam(weight, bias, weightGrad, biasGrad);
            adam.Step();
        }

        static void TestFromFunction()
        {
            float f(float x, float y) { return x*x + y*y; }
            float df(float x, float y) { return 2*x + 2*y; }

            float[,] weightArr = new float[2, 1];
            float[,] weightGradAr = new float[1000, 1000];
            float[] biasArr = new float[1000 * 1000];
            float[] biasGradArr = new float[1000 * 1000];
            for (int i = 0; i < 1000; i++)
            {
                for (int j = 0; j < 1000; j++)
                {
                    float x = ((float)i / 100f) - 5f;
                    float y = ((float)j / 100f) - 5f;
                    float fx = f(x, y);
                    float dfx = df(x, y);
                    weightArr[i, j] = fx;
                    weightGradAr[i, j] = dfx;
                    biasArr[j * (i + 1) + j] = fx;
                    biasGradArr[j * (i + 1) + j] = dfx;
                }
            }

            Matrix<float> weight = Matrix<float>.Build.DenseOfArray(weightArr);
            Matrix<float> weightGrad = Matrix<float>.Build.DenseOfArray(weightGradAr);

            Vector<float> bias = Vector<float>.Build.DenseOfArray(biasArr);
            Vector<float> biasGrad = Vector<float>.Build.DenseOfArray(biasGradArr);

            Adam adam = new Adam(weight, bias, weightGrad, biasGrad);
            adam.Step();
        }
    }
}
