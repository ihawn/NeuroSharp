using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.Datatypes
{
    public struct Tensor
    {
        public double[] Values { get; private set; }
        public int[] Shape { get; private set; }
        public int XShape { get { return Shape[0]; } }
        public int YShape { get { return Shape[1]; } }
        public int ZShape { get { return Shape[2]; } }

        public Tensor(int xSize, int ySize = 1, int zSize = 1)
        {
            Shape = new int[] { xSize, ySize, zSize };
            Values = new double[xSize * ySize * zSize];
        }

        public double this[int x, int y = 0, int z = 0]
        {
            get
            { 
                return Values[x + y * Shape[0] + z * Shape[0] * Shape[1]];
            }
            set
            {
                Values[x + y * Shape[0] + z * Shape[0] * Shape[1]] = value;
            }
        }

        public static Tensor operator *(Tensor t1, Tensor t2)
        {
            Tensor tOut = new Tensor(t1.YShape, t2.XShape, t1.ZShape);
            for (int z = 0; z < t1.ZShape; z++)
            {
                Parallel.For(0, t2.XShape, y =>
                {
                    for (int x = 0; x < t1.YShape; x++)
                        for (int n = 0; n < t1.XShape; n++)
                            tOut[x, y, z] += t1[n, y, z] * t2[x, n, z];
                });
            }
            return tOut;
        }

        public static Tensor operator *(Tensor t1, double d)
        {
            Tensor tOut = new Tensor(t1.YShape, t1.XShape, t1.ZShape);
            for (int z = 0; z < t1.ZShape; z++)
            {
                Parallel.For(0, t1.XShape, y =>
                {
                    for (int x = 0; x < t1.YShape; x++)
                        tOut[x, y, z] = t1[x, y, z] / d;
                });
            }
            return tOut;
        }

        public static Tensor operator /(Tensor t1, Tensor t2)
        {
            Tensor tOut = new Tensor(t1.XShape, t1.YShape, t1.ZShape);
            for (int z = 0; z < t1.ZShape; z++)
                Parallel.For(0, t1.XShape, x =>
                {
                    for (int y = 0; y < t1.YShape; y++)
                        tOut[x, y, z] = t1[x, y, z] / t2[x, y, z];
                });
            return tOut;
        }

        public static Tensor operator /(Tensor t1, double d)
        {
            Tensor tOut = new Tensor(t1.XShape, t1.YShape, t1.ZShape);
            for (int z = 0; z < t1.ZShape; z++)
                Parallel.For(0, t1.XShape, x =>
                {
                    for (int y = 0; y < t1.YShape; y++)
                        tOut[x, y, z] = t1[x, y, z] / d;
                });
            return tOut;
        }

        public static Tensor operator +(Tensor t1, Tensor t2)
        {
            Tensor tOut = new Tensor(t1.XShape, t1.YShape, t1.ZShape);
            for (int z = 0; z < t1.ZShape; z++)
                Parallel.For(0, t1.XShape, x =>
                {
                    for (int y = 0; y < t1.YShape; y++)
                        tOut[x, y, z] = t1[x, y, z] + t2[x, y, z];
                });
            return tOut;
        }

        public static Tensor operator -(Tensor t1, Tensor t2)
        {
            Tensor tOut = new Tensor(t1.XShape, t1.YShape, t1.ZShape);
            for (int z = 0; z < t1.ZShape; z++)
                Parallel.For(0, t1.XShape, x =>
                {
                    for (int y = 0; y < t1.YShape; y++)
                        tOut[x, y, z] = t1[x, y, z] - t2[x, y, z];
                });
            return tOut;
        }

        public Tensor PointwiseSquare()
        {
            Tensor tOut = new Tensor(XShape, YShape, ZShape);
            for(int z = 0; z < ZShape; z++)
                for(int x = 0; x < XShape; x++)
                    for (int y = 0; y < YShape; y++)
                        tOut[x, y, z] = this[x, y, z] * this[x, y, z];
            return tOut;
        }

        public Tensor PointwiseSqrt()
        {
            Tensor tOut = new Tensor(XShape, YShape, ZShape);
            for (int z = 0; z < ZShape; z++)
                for (int x = 0; x < XShape; x++)
                    for (int y = 0; y < YShape; y++)
                        tOut[x, y, z] = Math.Sqrt(this[x, y, z]);
            return tOut;
        }
    }
}
