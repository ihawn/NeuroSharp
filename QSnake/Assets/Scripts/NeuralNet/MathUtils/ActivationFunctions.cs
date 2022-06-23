using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;


namespace NeuroSharp
{
    public static class ActivationFunctions
    {
        public static float Tanh(float x)
        {
            return (float)Math.Tanh(x);
        }

        public static float TanhPrime(float x)
        {
            return 1 - (float)Math.Pow(Math.Tanh(x), 2);
        }

        public static float Relu(float x)
        {
            return Mathf.Max(0, x);
        }

        public static float ReluPrime(float x)
        {
            return x > 0 ? 1 : 0;
        }
    }
}
