using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

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

        public override Vector<float> BackPropagation(Vector<float> outputError)
        {
            Vector<float> vec = Vector<float>.Build.Dense(Input.Count);
            for (int i = 0; i < Input.Count; i++)
                vec[i] = ActivationPrime(Input[i]);
            var output = vec.PointwiseMultiply(outputError);
            return output;
        }
    }
}
