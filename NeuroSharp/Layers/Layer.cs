using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;
using NeuroSharp.Training;

namespace NeuroSharp
{
    public abstract class Layer
    {
        public int Id { get; set; }
        public Vector<double> Input { get; set; }
        public Vector<double> Output { get; set; }
        public int InputSize { get; set; }
        public int OutputSize { get; set; }
        public Network ParentNetwork { get; set; }
        public LayerType LayerType { get; set; }

        public abstract Vector<double> ForwardPropagation(Vector<double> input);
        public abstract Vector<double> BackPropagation(Vector<double> outputError);
        public virtual void SetSizeIO()
        {
            InputSize = Id > 0 ? ParentNetwork.Layers[Id - 1].OutputSize : ParentNetwork.EntrySize;
            OutputSize = OutputSize == 0 ? InputSize : OutputSize;
        }
        //todo: add automatically calculated input, output, parameter, etc size
    }
}
