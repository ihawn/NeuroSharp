using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuroSharp
{
    public static class ActivationFunctions
    {
        public static float Tanh(float x)
        {
            return MathF.Tanh(x);
        }

        public static float TanhPrime(float x)
        {
            return 1 - MathF.Pow(MathF.Tanh(x), 2);
        }
    }
}
