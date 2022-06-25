using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroSharp;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;

public class Training
{
    NeuroSharp.Network Network { get; set; }

    GameState GameState;
    int BatchSize { get; set; }
    int MaxQSize { get; set; }
    float DiscountRate { get; set; }
    float LearningRate { get; set; }
    float Err { get; set; }
    int DataCount { get; set; }

    //current state (11) | action (3) | reward | next state (11) | gameOver
    List<(Vector<float>, Vector<float>, float, Vector<float>, bool)> QTable { get; set; }

    public Training(GameState state, int batchSize = 1500)
    {
        GameState = state;
        BatchSize = batchSize;
        LearningRate = 0.1f;
        DiscountRate = 0.9f;
        QTable = new List<(Vector<float>, Vector<float>, float, Vector<float>, bool)>();
        MaxQSize = 100000;

        Network = new NeuroSharp.Network();
        Network.Add(new FullyConnectedLayer(11, 256));
        //Network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
        Network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
        Network.Add(new FullyConnectedLayer(256, 3));
        //Network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
        //Network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
        Network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);
    }

    public Direction GetNextDirection(Vector<float> state)
    {
        Vector<float> prediction = PredictQValues(state);
        return (Direction)prediction.MaximumIndex();
    }

    Vector<float> PredictQValues(Vector<float> state)
    {
        return Network.Predict(new List<Vector<float>> { state })[0];
    }

    public void UpdateNetwork(Vector<float> originState, Direction directionTaken, float rewardObtained, bool trainShortMemory, bool diedThisTurn)
    {
        //update game memory
        float[] directionArray = new float[3];
        directionArray[(int)directionTaken] = 1;
        Vector<float> categoricalDirectionTaken = Vector<float>.Build.DenseOfArray(directionArray);
        Vector<float> nextState = GameState.StateVector;

        var currentBigState = (originState, categoricalDirectionTaken, rewardObtained, nextState, diedThisTurn);
        QTable.Add(currentBigState);


        //get data to be trained on this step
        List<(Vector<float>, Vector<float>, float, Vector<float>, bool)> trainData;
        if (trainShortMemory)
        {
            trainData = new List<(Vector<float>, Vector<float>, float, Vector<float>, bool)>() { currentBigState };
        }
        else
        {
            System.Random r = new System.Random();
            trainData = QTable.OrderBy(x => r.Next()).Take(Mathf.Min(QTable.Count, BatchSize)).ToList();
            trainData.Add(currentBigState);

            Err /= DataCount;
            DataCount = 0;
            Debug.Log("Error: " + Err);
        }


        for (int i = 0; i < trainData.Count; i++)
        {
            var memoryObject = new { CurrentState = trainData[i].Item1, Action = trainData[i].Item2, Reward = trainData[i].Item3, NextState = trainData[i].Item4, GameOver = trainData[i].Item5 };
            float qNew = memoryObject.Reward;

            Vector<float> prediction = PredictQValues(memoryObject.CurrentState); //predict q value based on the current state in training data
            Vector<float> target = Vector<float>.Build.DenseOfVector(prediction); //clone prediction

            if (!memoryObject.GameOver)
            {
                float maxExpected = PredictQValues(memoryObject.NextState).Max();
                qNew = memoryObject.Reward + DiscountRate * maxExpected;
            }

            target[memoryObject.Action.MaximumIndex()] = qNew;
            Err += Network.Loss(target, prediction);
            DataCount++;

            //back propagation
            Vector<float> error = Network.LossPrime(target, prediction);
            for (int k = Network.Layers.Count - 1; k >= 0; k--)
            {
                error = Network.Layers[k].BackPropagation(error);
            }
        }
    }
}
