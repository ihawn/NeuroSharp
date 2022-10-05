using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.Datatypes
{
    public struct LSTMCellModel
    {
        public Vector<double> Input { get; set; }
        public Vector<double> Output { get; set; }
        public Vector<double> EmbeddingTransformation { get; set; }
        public Vector<double> EmbeddingCache { get; set; }
        public Vector<double>[] LSTMActivations { get; set; }
        public Vector<double> ActivationVector { get; set; }
        public Vector<double> CellVector { get; set; }
    }
}