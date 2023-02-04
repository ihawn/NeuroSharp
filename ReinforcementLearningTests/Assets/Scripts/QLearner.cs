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

    public QLearner(float decayConstant)
    {
        Network = new NeuroSharp.Training.Network(14);
        Network.Add(new FullyConnectedLayer(96));
        Network.Add(new ActivationLayer(ActivationType.Tanh));
        Network.Add(new FullyConnectedLayer(48));
        Network.Add(new ActivationLayer(ActivationType.Tanh));
        Network.Add(new FullyConnectedLayer(3));
        Network.Add(new ActivationLayer(ActivationType.Tanh));
        Network.UseLoss(LossType.MeanSquaredError);

        DecayConstant = decayConstant;
        Rewards = new List<QData>();
    }

    public Move GetNextMove(MoveResult lastMoveResult, float prevMoveReward, int prevScore, Vector<double> gameSpaceData)
    {
        if (lastMoveResult == MoveResult.Eat)
            for (int i = Rewards.Count - 1; i >= Rewards.Count - 7; i--)
                Rewards[i].Result[Rewards[i].Result.ToList().IndexOf(Rewards[i].Result.Max())] += 1;

        Debug.Log(Network.NetworkLoss);

        if (Iteration > 0)
            AddToRewardIndex(prevMoveReward, prevScore);

        Iteration++;
        previousGameSpaceData = gameSpaceData;

        if (GetRandomActionProbability() - 0.05 > Random.Range(0f, 1f))
        {
            Debug.Log("Random Move");
            previousMove = (Move)Random.Range(0, 3);
            return previousMove;
        }

        Vector<double> prediction = Network.Predict(gameSpaceData);
        previousMove = (Move)prediction.ToList().IndexOf(prediction.Max());
        return previousMove;
    }

    void AddToRewardIndex(float prevMoveReward, int prevScore)
    {
        Vector<double> result = Vector<double>.Build.Dense(3);
        result[(int)previousMove] = prevMoveReward;
        Rewards.Add(new QData(previousGameSpaceData, result, prevScore));
    }

    public void TrainStep(int take)
    {
        List<Vector<double>> xData = Rewards.Select(x => x.Data).Skip(Rewards.Count() - take).Take(take).ToList();
        List<Vector<double>> yData = Rewards.Select(x => x.Result).Skip(Rewards.Count() - take).Take(take).ToList();

        Network.Train(xData, yData, 1, TrainingConfiguration.SGD, OptimizerType.Adam, learningRate: 0.0004);
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
    public int Score { get; set; }

    public QData(Vector<double> data, Vector<double> result, int score) { Data = data; Result = result; Score = score;  }
}
