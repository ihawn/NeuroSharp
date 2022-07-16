using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class ActivationLayer : Layer
    {
        public Func<double, double> Activation { get; set; }
        public Func<double, double> ActivationPrime { get; set; }

        public ActivationLayer(Func<double, double> activation, Func<double, double> activationPrime)
        {
            Activation = activation;
            ActivationPrime = activationPrime;
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Output = Vector<double>.Build.Dense(input.Count);
            for(int i = 0; i < input.Count; i++)
                Output[i] = Activation(input[i]);

            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError, OptimizerType optimzerType, int sampleIndex, double learningRate = 0.001f)
        {
            Vector<double> vec = Vector<double>.Build.Dense(Input.Count);
            for (int i = 0; i < Input.Count; i++)
                vec[i] = ActivationPrime(Input[i]);
            return vec.PointwiseMultiply(outputError);
        }
    }
}
