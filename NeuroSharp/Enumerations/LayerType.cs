using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroSharp.Enumerations
{
    public enum LayerType
    {
        FullyConnected = 0,
        Convolutional = 1,
        Activation = 3,
        SoftmaxActivation = 4,
        MaxPooling = 5,
        Recurrent = 6
    }
}
