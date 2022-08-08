using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using NeuroSharp.Training;

namespace NeuroSharp.Model
{
    public class SerializeableNetwork
    {
        public List<string> Layers;

        public SerializeableNetwork(Network network)
        {
            Layers = new List<string>();
            foreach(var layer in network.Layers)
            {
                if (layer is ParameterizedLayer)
                    Layers.Add(JsonSerializer.Serialize(layer));
                else
                { 
                }
            }
        }
    }
}
