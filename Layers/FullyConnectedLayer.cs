using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class FullyConnectedLayer : Layer
    {
        public Matrix<float> Weights { get; set; }
        public Vector<float> Bias { get; set; }

        //Adam containers
        public Matrix<float> MeanWeightGradient { get; set; }
        public Vector<float> MeanBiasGradient { get; set; }
        public Matrix<float> VarianceWeightGradient { get; set; }
        public Vector<float> VarianceBiasGradientt { get; set; }

        public FullyConnectedLayer(int inputSize, int outputSize)
        {
            Weights = Matrix<float>.Build.Random(inputSize, outputSize);
            Bias = Vector<float>.Build.Random(outputSize);

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                    Weights[i, j] = MathUtils.Utils.NextFloat(-0.5f, 0.5f);
            for (int i = 0; i < outputSize; i++)
                Bias[i] = MathUtils.Utils.NextFloat(-0.5f, 0.5f);

            MeanWeightGradient = Matrix<float>.Build.Dense(inputSize, outputSize);
            MeanBiasGradient = Vector<float>.Build.Dense(outputSize);
            VarianceWeightGradient = Matrix<float>.Build.Dense(inputSize, outputSize);
            VarianceBiasGradientt = Vector<float>.Build.Dense(outputSize);
        }

        public override Vector<float> ForwardPropagation(Vector<float> input)
        {
            Input = input;
            Output = Input * Weights + Bias;
            return Output;
        }

        public override Vector<float> BackPropagation(Vector<float> outputError, OptimizerType optimzerType, int sampleIndex, float learningRate)
        {
            Vector<float> inputError = outputError * Weights.Transpose();
            Matrix<float> weightsError = Input.OuterProduct(outputError);

            switch (optimzerType)
            {
                case OptimizerType.GradientDescent:
                    Weights -= learningRate * weightsError;
                    Bias -= learningRate * outputError;
                    break;

                case OptimizerType.Adam:
                    AdamOutput a = Adam.Step(Weights, Bias, weightsError, outputError, sampleIndex + 1, MeanWeightGradient, MeanBiasGradient, VarianceWeightGradient, VarianceBiasGradientt, eta: learningRate);
                    Weights = a.Weights;
                    Bias = a.Bias;
                    MeanWeightGradient = a.MeanWeightGradient;
                    MeanBiasGradient = a.MeanBiasGradient;
                    VarianceWeightGradient = a.VarianceWeightGradient;
                    VarianceBiasGradientt = a.VarianceBiasGradient;
                    break;
            }

            return inputError;
        }
    }
}
