using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.Datatypes
{
    public struct LSTMCellModel
    {
        public Vector<double> Input { get; set; }
        public Vector<double>[] LSTMActivations { get; set; }
        public Vector<double> FlattenedActivationMatrix { get; set; }
        public Vector<double> FlattenedCellMatrix { get; set; }
        public Vector<double> Output { get; set; }
    }
}