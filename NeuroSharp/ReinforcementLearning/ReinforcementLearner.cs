using System;
using  MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Models;
using NeuroSharp.Training;
using System.Collections.Generic;
using NeuroSharp.Enumerations;
using System.Linq;

namespace NeuroSharp.ReinforcementLearning
{
    public class ReinforcementLearner<T>
    {
        public Network Network { get; set; }
        public int Iteration { get; set; }
        public double DecayConstant { get; set; }
        public List<LearnerData> Rewards { get; set; }
        public TrainingConfiguration TrainingConfiguration { get; set; }
        public OptimizerType OptimizerType { get; set; }
        public double LearningRate { get; set; }
        public int PositiveReinforcementLookback { get; set; }
        
        private T previousMove;
        private Vector<double> previousSpaceData;

        public ReinforcementLearner(Network network, TrainingConfiguration config, OptimizerType type, 
            double learningRate, double decayConstant, int lookback)
        {
            Network = network;
            DecayConstant = decayConstant;
            TrainingConfiguration = config;
            OptimizerType = type;
            LearningRate = learningRate;
            PositiveReinforcementLookback = lookback;
            Rewards = new List<LearnerData>();
        }
        
        public T GetNextMove(int lastMoveResult, float prevMoveReward, int prevScore, List<T> possibleActions, Vector<double> gameSpaceData)
        {
            if (lastMoveResult == 0)
                for (int i = Rewards.Count - 1; i >= Rewards.Count - PositiveReinforcementLookback; i--)
                    Rewards[i].Result[Rewards[i].Result.ToList().IndexOf(Rewards[i].Result.Max())] += 1;

            if (Iteration > 0)
                AddToRewardIndex(prevMoveReward, prevScore, possibleActions.Count, possibleActions.IndexOf(previousMove));

            Iteration++;
            previousSpaceData = gameSpaceData;

            if (GetRandomActionProbability() - 0.05 > new Random().NextDouble())
            {
                previousMove = possibleActions[new Random().Next(possibleActions.Count)];
                return previousMove;
            }

            Vector<double> prediction = Network.Predict(gameSpaceData);
            previousMove = possibleActions[prediction.ToList().IndexOf(prediction.Max())];
            return previousMove;
        }
        
        public void TrainStep(int take)
        {
            List<Vector<double>> xData = Rewards.Select(x => x.Data).Skip(Rewards.Count() - take).Take(take).ToList();
            List<Vector<double>> yData = Rewards.Select(x => x.Result).Skip(Rewards.Count() - take).Take(take).ToList();

            Network.Train(xData, yData, 1, TrainingConfiguration, OptimizerType, learningRate: LearningRate);
        }
        
        void AddToRewardIndex(float prevMoveReward, int prevScore, int actionCount, int moveIndex)
        {
            Vector<double> result = Vector<double>.Build.Dense(actionCount);
            result[moveIndex] = prevMoveReward;
            Rewards.Add(new LearnerData(previousSpaceData, result, prevScore));
        }
        
        double GetRandomActionProbability()
        {
            return Math.Exp(-DecayConstant * Iteration);
        }
    }
}