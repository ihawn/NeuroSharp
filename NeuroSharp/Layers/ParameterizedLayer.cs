using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public abstract class ParameterizedLayer : Layer
    {
        public Matrix<float> Weights { get; set; }
        public Vector<float> Bias { get; set; }
        public Matrix<float> WeightGradient { get; set; }
        public Vector<float> BiasGradient { get; set; }
    }
}
