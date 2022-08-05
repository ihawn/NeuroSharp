using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public abstract class Layer
    {
        public Vector<double> Input { get; set; }
        public Vector<double> Output { get; set; }

        public abstract Vector<double> ForwardPropagation(Vector<double> input);
        public abstract Vector<double> BackPropagation(Vector<double> outputError);
    }
}
