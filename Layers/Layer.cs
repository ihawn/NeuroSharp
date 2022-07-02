using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Optimizers;

namespace NeuroSharp
{
    public abstract class Layer
    {
        public Vector<float> Input { get; set; }
        public Vector<float> Output { get; set; }

        public abstract Vector<float> ForwardPropagation(Vector<float> input);
        public abstract Vector<float> BackPropagation(Vector<float> outputError, float learningRate = 0.001f, int sampleIndex = 1);
    }
}
