using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroSharp;
using NeuroSharp.Enumerations;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;

public class QLearner : MonoBehaviour
{
    public NeuroSharp.Training.Network Network { get; private set; }
    public SnakeGame SnakeGame { get; private set; }
    public int Iteration { get; set; }
    public float DecayConstant { get; set; }
    public List<QData> Rewards { get; set; }

    private float gamma = 0.9f;
    private float alpha = 0.2f;
    private Move previousMove;
    private Vector<double> previousGameSpaceData;

    public QLearner(SnakeGame snakeGame, float decayConstant)
    {
        Network = new NeuroSharp.Training.Network(3);
        Network.Add(new FullyConnectedLayer(64));
        Network.Add(new ActivationLayer(ActivationType.Tanh));
        Network.Add(new FullyConnectedLayer(3));
        Network.Add(new ActivationLayer(ActivationType.Tanh));
        Network.UseLoss(LossType.MeanSquaredError);

        DecayConstant = decayConstant;
        Rewards = new List<QData>();
    }

    public Move GetNextMove(float prevMoveReward, Vector<double> gameSpaceData)
    {
        TrainStep();
        Debug.Log(Network.NetworkLoss);

        if (Iteration > 0)
            AddToRewardIndex(prevMoveReward);

        Iteration++;
        previousGameSpaceData = gameSpaceData;

        if (GetRandomActionProbability() > Random.Range(0f, 1f))
        {
            previousMove = (Move)Random.Range(0, 3);
            return previousMove;
        }

        Vector<double> prediction = Network.Predict(gameSpaceData);
        previousMove = (Move)prediction.ToList().IndexOf(prediction.Max());
        return previousMove;
    }

    void AddToRewardIndex(float prevMoveReward)
    {
        Vector<double> result = Vector<double>.Build.Dense(3);
        result[(int)previousMove] = prevMoveReward;
        Rewards.Add(new QData(previousGameSpaceData, result));
    }

    void TrainStep()
    {
        Rewards = Rewards.OrderBy(x => Random.Range(0f, 1f)).ToList();
        List<Vector<double>> xData = Rewards.Select(x => x.Data).Take(200).ToList();
        List<Vector<double>> yData = Rewards.Select(x => x.Result).Take(200).ToList();

        Network.Train(xData, yData, 1, TrainingConfiguration.SGD, OptimizerType.Adam, learningRate: 0.001);
    }

    double GetRandomActionProbability()
    {
        return Mathf.Exp(-DecayConstant * Iteration);
    }
}

public struct QData
{
    public Vector<double> Data { get; set; }
    public Vector<double> Result { get; set; }

    public QData(Vector<double> data, Vector<double> result) { Data = data; Result = result; }
}
