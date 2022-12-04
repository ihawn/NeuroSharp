using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;
using NeuroSharp.Training;
using Newtonsoft.Json;

namespace NeuroSharp
{
    //todo: try to replace all flatten and unflatten with implicit indexing (maybe profile first to see if this is necessary)
    public abstract class Layer //todo: dropout layer
    {
        public int Id { get; set; }
        
        [JsonIgnore]
        public Vector<double> Input { get; set; }
        [JsonIgnore]
        public Vector<double> Output { get; set; }
        public int InputSize { get; set; }
        public int OutputSize { get; set; }
        
        [JsonIgnore]
        public Network ParentNetwork { get; set; }
        public LayerType LayerType { get; set; }

        public abstract Vector<double> ForwardPropagation(Vector<double> input);
        public abstract Vector<double> BackPropagation(Vector<double> outputError);
        public virtual void SetSizeIO()
        {
            InputSize = Id > 0 ? ParentNetwork.Layers[Id - 1].OutputSize : ParentNetwork.EntrySize;
            OutputSize = OutputSize == 0 ? InputSize : OutputSize;
        }
    }
}
