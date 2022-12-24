using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using ModuleHandle = UnityEngine.U2D.Common.UTess.ModuleHandle;
using Unity.Burst;
using Unity.Jobs;

namespace UnityEditor.U2D.Animation
{

    // Biharmonics In Jobs.
    [BurstCompile]
    internal struct BiharmonicsJob : IJob
    {

        // Output.
        public int numIterations;
        public int numSamples;
        public int suppressCommonWarnings;
        public NativeArray<BoneWeight> weights;

        // Input
        [DeallocateOnJobCompletion]
        public NativeArray<float2> inputVertices;
        [DeallocateOnJobCompletion]
        public NativeArray<int> inputIndices;
        [DeallocateOnJobCompletion]
        public NativeArray<int2> inputEdges;
        [DeallocateOnJobCompletion]
        public NativeArray<float2> inputControlPoints;
        [DeallocateOnJobCompletion]
        public NativeArray<int2> inputBones;
        [DeallocateOnJobCompletion]
        public NativeArray<int> inputPins;

        [BurstCompile]
        public void Execute()
        {
            unsafe
            {
                // Run the Biharmonics.
                BoundedBiharmonicWeightsGenerator.CalculateInternal_((float2*)inputVertices.GetUnsafePtr(), inputVertices.Length, 
                    (int*)inputIndices.GetUnsafePtr(), inputIndices.Length, 
                    (float2*)inputControlPoints.GetUnsafePtr(), inputControlPoints.Length, 
                    (int2*)inputBones.GetUnsafePtr(), inputBones.Length, 
                    (int*)inputPins.GetUnsafePtr(), inputPins.Length, 
                    numSamples, numIterations, suppressCommonWarnings, 
                    (BoneWeight*)weights.GetUnsafePtr(), weights.Length);
            }
        }

    }

    [BurstCompile]
    internal class BoundedBiharmonicWeightsGenerator : IWeightsGenerator
    {
        internal static readonly BoneWeight defaultWeight = new BoneWeight() { weight0 = 1 };
        internal static readonly int k_NumIterations = 100;
        internal static readonly int k_NumSamples = 4;

        [DllImport("BoundedBiharmonicWeightsModule")]
        static extern int Bbw(int iterations,
            [In, Out] IntPtr vertices, int vertexCount, int originalVertexCount,
            [In, Out] IntPtr indices, int indexCount,
            [In, Out] IntPtr controlPoints, int controlPointsCount,
            [In, Out] IntPtr boneEdges, int boneEdgesCount,
            [In, Out] IntPtr pinIndices, int pinIndexCount,
            [In, Out] IntPtr weights
            );

        public BoneWeight[] Calculate(string name, in float2[] vertices, in int[] indices, in int2[] edges, in float2[] controlPoints, in int2[] bones, in int[] pins)
        {
            // In almost all cases subdivided mesh weights fine. Non-subdivide is only a fail-safe.
            var success = false;
            var sanitizedEdges = SanitizeEdges(edges, vertices.Length);
            var weights = CalculateInternal(vertices, indices, sanitizedEdges, controlPoints, bones, pins, k_NumSamples, ref success);
            return weights;
        }

        public JobHandle CalculateJob(string name, in float2[] vertices, in int[] indices, in int2[] edges, in float2[] controlPoints, in int2[] bones, in int[] pins, SpriteJobData sd)
        {
            // In almost all cases subdivided mesh weights fine. Non-subdivide is only a fail-safe.
            var sanitizedEdges = SanitizeEdges(edges, vertices.Length);
            var bbwJob = new BiharmonicsJob();
            bbwJob.weights = sd.weights;
            bbwJob.inputVertices = new NativeArray<float2>(vertices, Allocator.TempJob);
            bbwJob.inputIndices = new NativeArray<int>(indices, Allocator.TempJob);
            bbwJob.inputEdges = new NativeArray<int2>(sanitizedEdges, Allocator.TempJob);
            bbwJob.inputControlPoints = new NativeArray<float2>(controlPoints, Allocator.TempJob);
            bbwJob.inputBones = new NativeArray<int2>(bones, Allocator.TempJob);
            bbwJob.inputPins = new NativeArray<int>(pins, Allocator.TempJob);
            bbwJob.numSamples = k_NumSamples;
            bbwJob.numIterations = k_NumIterations;
            bbwJob.suppressCommonWarnings = PlayerSettings.suppressCommonWarnings ? 1 : 0;
            return bbwJob.Schedule();
        }

        static int2[] SanitizeEdges(in int2[] edges, int noOfVertices)
        {
            var tmpEdges = new List<int2>(edges);
            for (var i = tmpEdges.Count - 1; i >= 0; i--)
            {
                if (tmpEdges[i].x >= noOfVertices || tmpEdges[i].y >= noOfVertices)
                    tmpEdges.RemoveAt(i);
            }

            return tmpEdges.ToArray();
        }

        [BurstCompile]
        static unsafe void SampleBones_(
            float2* pointsPtr, 
            int2* edgesPtr, int edgesCount, 
            int numSamples, 
            float2* sampledEdgesPtr)
        {
            Debug.Assert(numSamples > 0);

            var j = 0;
            for (var i = 0; i < edgesCount; i++)
            {
                var edge = edgesPtr[i];
                var tip = pointsPtr[edge.x];
                var tail = pointsPtr[edge.y];

                for (var s = 0; s < numSamples; s++)
                {
                    var f = (s + 1f) / (float)(numSamples + 1f);
                    sampledEdgesPtr[j++] = f * tail + (1f - f) * tip;
                }
            }
        }

        [BurstCompile]
        // Triangulate Bone Samplers.
        static unsafe void TriangulateSamplers_(
            float2* samplersPtr, int samplerCount, 
            float2* triVerticesPtr, 
            ref int vtxCount, 
            int* triIndicesPtr, 
            ref int idxCount)
        {
            for (var i = 0; i < samplerCount; ++i)
            {
                var v = samplersPtr[i];
                var vtxCount_ = vtxCount;
                var triCount_ = idxCount / 3;

                for (var m = 0; m < triCount_; ++m)
                {
                    var i1 = triIndicesPtr[0 + (m * 3)];
                    var i2 = triIndicesPtr[1 + (m * 3)];
                    var i3 = triIndicesPtr[2 + (m * 3)];
                    var v1 = triVerticesPtr[i1];
                    var v2 = triVerticesPtr[i2];
                    var v3 = triVerticesPtr[i3];
                    var inside = ModuleHandle.IsInsideTriangle(v, v1, v2, v3);
                    if (inside)
                    {
                        triVerticesPtr[vtxCount] = v;
                        triIndicesPtr[idxCount++] = i1; triIndicesPtr[idxCount++] = i2; triIndicesPtr[idxCount++] = vtxCount;
                        triIndicesPtr[idxCount++] = i2; triIndicesPtr[idxCount++] = i3; triIndicesPtr[idxCount++] = vtxCount;
                        triIndicesPtr[idxCount++] = i3; triIndicesPtr[idxCount++] = i1; triIndicesPtr[idxCount++] = vtxCount;
                        vtxCount++;
                        break;
                    }
                }
            }
        }

        [BurstCompile]
        // Triangulate Skipped Original Points. These points are discarded during PlanarGrapg cleanup. But bbw only cares if these are part of any geometry. So just insert them.
        static unsafe void TriangulateInternal_(void* internalIndicesPtr, 
            int internalIndexCount, 
            void* triVerticesPtr, 
            void* triIndicesPtr, 
            ref int idxCount)
        {
            var internalIndices = (int*) internalIndicesPtr;
            var triVertices = (float2*) triVerticesPtr;
            var triIndices = (int*) triIndicesPtr;
            
            var triangleCount = idxCount / 3;

            for (var j = 0; j < internalIndexCount; ++j)
            {
                var index = internalIndices[j];
                var v = triVertices[index];
                for (var i = 0; i < triangleCount; ++i)
                {
                    var i1 = triIndices[0 + (i * 3)];
                    var i2 = triIndices[1 + (i * 3)];
                    var i3 = triIndices[2 + (i * 3)];
                    var v1 = triVertices[i1];
                    var v2 = triVertices[i2];
                    var v3 = triVertices[i3];
                    var c1 = (float)Math.Round(ModuleHandle.OrientFast(v1, v2, v), 2);
                    if (c1 == 0)
                    {
                        triIndices[0 + (i * 3)] = i1; triIndices[1 + (i * 3)] = index; triIndices[2 + (i * 3)] = i3;
                        triIndices[idxCount++] = index; triIndices[idxCount++] = i2; triIndices[idxCount++] = i3;
                    }
                    else
                    {
                        var c2 = (float)Math.Round(ModuleHandle.OrientFast(v2, v3, v), 2);
                        if (c2 == 0)
                        {
                            triIndices[0 + (i * 3)] = i2; triIndices[1 + (i * 3)] = index; triIndices[2 + (i * 3)] = i1;
                            triIndices[idxCount++] = index; triIndices[idxCount++] = i3; triIndices[idxCount++] = i1;
                        }
                        else
                        {
                            var c3 = (float)Math.Round(ModuleHandle.OrientFast(v3, v1, v), 2);
                            if (c3 == 0)
                            {
                                triIndices[0 + (i * 3)] = i3; triIndices[1 + (i * 3)] = index; triIndices[2 + (i * 3)] = i2;
                                triIndices[idxCount++] = index; triIndices[idxCount++] = i1; triIndices[idxCount++] = i2;
                            }
                        }
                    }
                }
            }
        }

        [BurstCompile]
        internal static unsafe void CalculateInternal_(float2* inputVerticesPtr, int inputVertexCount, 
            int* inputIndicesPtr, int inputIndicesCount,
            float2* inputControlPointsPtr, int inputControlPointCount, 
            int2* inputBonesPtr, int inputBoneCount, 
            int* inputPinsPtr, int inputPinCount, 
            int numSamples, int iterations, int suppressWarnings, 
            BoneWeight* weightPtr, int weightCount)
        {
            for (var i = 0; i < weightCount; ++i)
                weightPtr[i] = defaultWeight;
            if (inputVertexCount < 3)
                return;
            
            var boneSamples = new NativeArray<float2>(inputBoneCount * numSamples, Allocator.Temp);
            SampleBones_(inputControlPointsPtr,
                inputBonesPtr, inputBoneCount, 
                numSamples, 
                (float2*)boneSamples.GetUnsafePtr());

            // Copy Original Indices.
            var cntIndices = 0;
            var indicesCnt = 8 * (inputIndicesCount + inputControlPointCount + inputBoneCount);
            var tmpIndices = new NativeArray<int>(indicesCnt, Allocator.Temp);
            for (var i = 0; i < inputIndicesCount / 3; ++i)
            {
                var i1 = inputIndicesPtr[0 + (i * 3)];
                var i2 = inputIndicesPtr[1 + (i * 3)];
                var i3 = inputIndicesPtr[2 + (i * 3)];
                var v1 = inputVerticesPtr[i1];
                var v2 = inputVerticesPtr[i2];
                var v3 = inputVerticesPtr[i3];
                var rt = (float)Math.Round(ModuleHandle.OrientFast(v1, v2, v3), 2);
                if (rt != 0)
                {
                    tmpIndices[cntIndices++] = i1;
                    tmpIndices[cntIndices++] = i2;
                    tmpIndices[cntIndices++] = i3;
                }
            }

            // Insert Samplers.
            var internalPoints = new NativeArray<int>(inputVertexCount, Allocator.Temp);
            var internalPointsCnt = 0;
            for (var i = 0; i < inputVertexCount; ++i)
            {
                var counter = 0;
                for (var m = 0; m < cntIndices; ++m)
                {
                    if (tmpIndices[m] == i)
                        counter++;
                }
                if (counter == 0)
                    internalPoints[internalPointsCnt++] = i;
            }
            TriangulateInternal_(internalPoints.GetUnsafePtr(), 
                internalPointsCnt, 
                inputVerticesPtr, 
                tmpIndices.GetUnsafePtr(), 
                ref cntIndices);

            var cntVertices = 0;
            var verticesCnt = 8 * (inputVertexCount + boneSamples.Length + inputControlPointCount);
            var tmpVertices = new NativeArray<float2>(verticesCnt, Allocator.Temp);
            for (var i = 0; i < inputVertexCount; i++)
                tmpVertices[cntVertices++] = inputVerticesPtr[i];

            TriangulateSamplers_((float2*)boneSamples.GetUnsafePtr(), boneSamples.Length, 
                (float2*)tmpVertices.GetUnsafePtr(), 
                ref cntVertices, 
                (int*)tmpIndices.GetUnsafePtr(), 
                ref cntIndices);
            
            TriangulateSamplers_(inputControlPointsPtr, inputControlPointCount, 
                (float2*)tmpVertices.GetUnsafePtr(), 
                ref cntVertices, 
                (int*)tmpIndices.GetUnsafePtr(), 
                ref cntIndices);

            var result = Bbw(iterations, (IntPtr)tmpVertices.GetUnsafePtr(), cntVertices, inputVertexCount,
                            (IntPtr)tmpIndices.GetUnsafePtr(), cntIndices,
                            (IntPtr)inputControlPointsPtr, inputControlPointCount,
                            (IntPtr)inputBonesPtr, inputBoneCount,
                            (IntPtr)inputPinsPtr, inputPinCount,
                            (IntPtr)weightPtr);

            switch (result)
            {
                case 1:
                case 2:
                    Debug.LogWarning($"Weight generation failure due to unexpected mesh input. Re-generate the mesh with a different Outline Detail value to resolve the issue. Error Code: {result}");
                    break;
                case 3:
                    if (0 == suppressWarnings)
                        Debug.LogWarning($"Internal weight generation error. Error Code: {result}");
                    break;
            }

            var done = false;
            for (var i = 0; i < weightCount; ++i)
            {
                var weight = weightPtr[i];

                if (weight.Sum() == 0f)
                    weightPtr[i] = defaultWeight;
                else if (!done)
                    done = (weight.boneIndex0 != 0 || weight.boneIndex1 != 0 || weight.boneIndex2 != 0 || weight.boneIndex3 != 0);
            }
        }

        static BoneWeight[] CalculateInternal(in float2[] inputVertices, in int[] inputIndices, in int2[] inputEdges, in float2[] inputControlPoints, in int2[] inputBones, in int[] inputPins, int numSamples, ref bool done)
        {

            var bbwJob = new BiharmonicsJob();
            bbwJob.weights = new NativeArray<BoneWeight>(inputVertices.Length, Allocator.Persistent);
            bbwJob.inputVertices = new NativeArray<float2>(inputVertices, Allocator.TempJob);
            bbwJob.inputIndices = new NativeArray<int>(inputIndices, Allocator.TempJob);
            bbwJob.inputEdges = new NativeArray<int2>(inputEdges, Allocator.TempJob);
            bbwJob.inputControlPoints = new NativeArray<float2>(inputControlPoints, Allocator.TempJob);
            bbwJob.inputBones = new NativeArray<int2>(inputBones, Allocator.TempJob);
            bbwJob.inputPins = new NativeArray<int>(inputPins, Allocator.TempJob);
            bbwJob.numSamples = numSamples;
            bbwJob.numIterations = k_NumIterations;
            bbwJob.suppressCommonWarnings = PlayerSettings.suppressCommonWarnings ? 1 : 0;
            var bbwJobHandle = bbwJob.Schedule();
            bbwJobHandle.Complete();
            var weights = bbwJob.weights.ToArray();
            bbwJob.weights.Dispose();
            return weights;
        }

        /*
        static BoneWeight[] CalculateInternal(in float2[] inputVertices, in int[] inputIndices, in int2[] inputEdges, in float2[] inputControlPoints, in int2[] inputBones, in int[] inputPins, int numSamples, ref bool done)
        {

            var bbwJob = new BiharmonicsJob();
            bbwJob.weights_ = new NativeArray<BoneWeight>(inputVertices.Length, Allocator.Persistent);
            bbwJob.inputVertices_ = new NativeArray<float2>(inputVertices, Allocator.TempJob);
            bbwJob.inputIndices_ = new NativeArray<int>(inputIndices, Allocator.TempJob);
            bbwJob.inputEdges_ = new NativeArray<int2>(inputEdges, Allocator.TempJob);
            bbwJob.inputControlPoints_ = new NativeArray<float2>(inputControlPoints, Allocator.TempJob);
            bbwJob.inputBones_ = new NativeArray<int2>(inputBones, Allocator.TempJob);
            bbwJob.inputPins_ = new NativeArray<int>(inputPins, Allocator.TempJob);
            bbwJob.numSamples_ = numSamples;
            bbwJob.numIterations_ = k_NumIterations;
            var bbwJobHandle = bbwJob.Schedule();
            bbwJobHandle.Complete();
            var weights = bbwJob.weights_.ToArray();
            bbwJob.weights_.Dispose();
            return weights;

            var controlPoints = EditorUtilities.CreateCopy(inputControlPoints);
            var vertices = EditorUtilities.CreateCopy(inputVertices);
            var indices = EditorUtilities.CreateCopy(inputIndices);

            var boneSamples = SampleBones(in inputControlPoints, in inputBones, numSamples);
            Round(ref vertices);
            Round(ref controlPoints);
            Round(ref boneSamples);

            // Copy Original Indices.
            var coIndices = EditorUtilities.CreateCopy(indices);
            var tmpIndices = new List<int>(indices.Length);
            for (var i = 0; i < coIndices.Length / 3; ++i)
            {
                var i1 = coIndices[0 + (i * 3)];
                var i2 = coIndices[1 + (i * 3)];
                var i3 = coIndices[2 + (i * 3)];
                var v1 = vertices[i1];
                var v2 = vertices[i2];
                var v3 = vertices[i3];
                var rt = (float)Math.Round(ModuleHandle.OrientFast(v1, v2, v3), 2);
                if (rt != 0)
                {
                    tmpIndices.Add(i1);
                    tmpIndices.Add(i2);
                    tmpIndices.Add(i3);
                }
            }
            indices = tmpIndices.ToArray();

            // Insert Samplers.
            var internalPoints = new List<int>();
            for (var i = 0; i < vertices.Length; ++i)
            {
                var counter = 0;
                for (var m = 0; m < indices.Length; ++m)
                {
                    if (indices[m] == i)
                        counter++;
                }
                if (counter == 0)
                    internalPoints.Add(i);
            }

            tmpIndices = new List<int>(indices);
            TriangulationUtility.TriangulateInternal(internalPoints.ToArray(), in vertices, ref tmpIndices);

            var tmpVertices = new List<float2>(vertices);
            TriangulationUtility.TriangulateSamplers(boneSamples, ref tmpVertices, ref tmpIndices);
            TriangulationUtility.TriangulateSamplers(controlPoints, ref tmpVertices, ref tmpIndices);
            vertices = tmpVertices.ToArray();
            indices = tmpIndices.ToArray();

            var verticesHandle = GCHandle.Alloc(vertices, GCHandleType.Pinned);
            var indicesHandle = GCHandle.Alloc(indices, GCHandleType.Pinned);
            var controlPointsHandle = GCHandle.Alloc(controlPoints, GCHandleType.Pinned);
            var bonesHandle = GCHandle.Alloc(inputBones, GCHandleType.Pinned);
            var pinsHandle = GCHandle.Alloc(inputPins, GCHandleType.Pinned);
            var weightsHandle = GCHandle.Alloc(weights, GCHandleType.Pinned);

            var result = Bbw(k_NumIterations,
                verticesHandle.AddrOfPinnedObject(), vertices.Length, inputVertices.Length,
                indicesHandle.AddrOfPinnedObject(), indices.Length,
                controlPointsHandle.AddrOfPinnedObject(), controlPoints.Length,
                bonesHandle.AddrOfPinnedObject(), inputBones.Length,
                pinsHandle.AddrOfPinnedObject(), inputPins.Length,
                weightsHandle.AddrOfPinnedObject());

            switch (result)
            {
                case 1:
                case 2:
                    Debug.LogWarning($"Weight generation failure due to unexpected mesh input. Re-generate the mesh with a different Outline Detail value to resolve the issue. Error Code: {result}");
                    break;
                case 3:
                    if (!PlayerSettings.suppressCommonWarnings)
                        Debug.LogWarning($"Internal weight generation error. Error Code: {result}");
                    break;
            }

            verticesHandle.Free();
            indicesHandle.Free();
            controlPointsHandle.Free();
            bonesHandle.Free();
            pinsHandle.Free();
            weightsHandle.Free();

            for (var i = 0; i < weights.Length; ++i)
            {
                var weight = weights[i];

                if (weight.Sum() == 0f)
                    weights[i] = defaultWeight;
                else if (!done)
                    done = (weight.boneIndex0 != 0 || weight.boneIndex1 != 0 || weight.boneIndex2 != 0 || weight.boneIndex3 != 0);
            }

            return weights;
        }

        static float2[] SampleBones(in float2[] points, in int2[] edges, int numSamples)
        {
            Debug.Assert(numSamples > 0);

            var sampledEdges = new List<float2>(edges.Length * numSamples);
            for (var i = 0; i < edges.Length; i++)
            {
                var edge = edges[i];
                var tip = points[edge.x];
                var tail = points[edge.y];

                for (var s = 0; s < numSamples; s++)
                {
                    var f = (s + 1f) / (float)(numSamples + 1f);
                    sampledEdges.Add(f * tail + (1f - f) * tip);
                }
            }

            return sampledEdges.ToArray();
        }

        static void Round(ref float2[] data)
        {
            for (var i = 0; i < data.Length; ++i)
            {
                var x = data[i].x;
                var y = data[i].y;
                x = (float)Math.Round(x, 8);
                y = (float)Math.Round(y, 8);
                data[i] = new float2(x, y);
            }
        }

        public void DebugMesh(ISpriteMeshData spriteMeshData, float2[] vertices, int2[] edges, float2[] controlPoints, int2[] bones, int[] pins)
        {
            var boneSamples = SampleBones(controlPoints, bones, k_NumSamples);
            var testVertices = new float2[vertices.Length + controlPoints.Length + boneSamples.Length];

            var headIndex = 0;
            Array.Copy(vertices, 0, testVertices, headIndex, vertices.Length);
            headIndex = vertices.Length;
            Array.Copy(controlPoints, 0, testVertices, headIndex, controlPoints.Length);
            headIndex += controlPoints.Length;
            Array.Copy(boneSamples, 0, testVertices, headIndex, boneSamples.Length);

            TriangulationUtility.Triangulate(ref edges, ref testVertices, out var indices, Allocator.Temp);


            spriteMeshData.Clear();

            for (var i = 0; i < testVertices.Length; ++i)
                spriteMeshData.AddVertex(testVertices[i], new BoneWeight());

            var convertedEdges = new Vector2Int[edges.Length];
            Array.Copy(edges, convertedEdges, edges.Length);
            spriteMeshData.edges = convertedEdges;
            spriteMeshData.indices = indices;
        }
        */
    }
}
