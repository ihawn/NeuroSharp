using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class ActivationLayer : Layer
    {
        public Func<float, float> Activation { get; set; }
        public Func<float, float> ActivationPrime { get; set; }

        public ActivationLayer(Func<float, float> activation, Func<float, float> activationPrime)
        {
            Activation = activation;
            ActivationPrime = activationPrime;
        }

        public override Vector<float> ForwardPropagation(Vector<float> input)
        {
            Input = input;
            Output = Vector<float>.Build.Dense(input.Count);
            for(int i = 0; i < input.Count; i++)
                Output[i] = Activation(input[i]);

            return Output;
        }

        public override Vector<float> BackPropagation(Vector<float> outputError, OptimizerType optimzerType, int sampleIndex, float learningRate = 0.001f)
        {
            Vector<float> vec = Vector<float>.Build.Dense(Input.Count);
            for (int i = 0; i < Input.Count; i++)
                vec[i] = ActivationPrime(Input[i]);
            return vec.PointwiseMultiply(outputError);
        }
    }
}
