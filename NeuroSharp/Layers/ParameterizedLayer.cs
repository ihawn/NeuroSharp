using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public abstract class ParameterizedLayer : Layer
    {
        public Matrix<double>[] Weights { get; set; }
        public Vector<double> Bias { get; set; }
        public Matrix<double>[] WeightGradient { get; set; }
        public Vector<double> BiasGradient { get; set; }
    }
}
