using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public abstract class Layer
    {
        public Vector<float> Input { get; set; }
        public Vector<float> Output { get; set; }

        public abstract Vector<float> ForwardPropagation(Vector<float> input);
        public abstract Vector<float> BackPropagation(Vector<float> outputError, OptimizerType optimzerType, int sampleIndex, float learningRate = 0.001f);
    }
}
