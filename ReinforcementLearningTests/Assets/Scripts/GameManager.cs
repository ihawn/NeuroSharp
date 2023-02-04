using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.ReinforcementLearning;
using NeuroSharp.Training;
using System;
using System.Linq;

public class GameManager : MonoBehaviour
{
    public SnakeGame SnakeGame;
    public GameObject SquareObject;
    public ReinforcementLearner<Move> Learner;
    public int GameSizeX = 6;
    public int GameSizeY = 6;
    public int SnakeStartLength = 3;

    public float PositiveMoveReward = 1;
    public float NeutralMoveReward = 0.2f;
    public float NegativeMoveReward = -1;
    public float CurrentMoveReward = 0;

    public float TimeStep = 1;
    public float MoveTimer = 0;
    public float GameTick = 1;

    public int MovesWithoutEating = 0;
    public int MovesWithoutEatingLimit = 20;

    public int Score;

    public MoveResult LastMoveResult;

    [Header("Obstacle Detection")]
    public int ObstacleLeft;
    public int ObstacleStraight;
    public int ObstacleRight;

    [Header("Orientation")]
    public int Up;
    public int Right;
    public int Down;
    public int Left;

    [Header("Food Direction")]
    public int FoodUp;
    public int FoodRight;
    public int FoodDown;
    public int FoodLeft;

    [Header("Food Adjacent Direction")]
    public int FoodDirectlyLeft;
    public int FoodDirectlyStraight;
    public int FoodDirectlyRight;

    private Vector<double> FlattenedGameSpaceData;

    Snake Snake { get { return SnakeGame.Snake; } }

    void Start()
    {
        SnakeGame = new SnakeGame(GameSizeX, GameSizeY, SnakeStartLength, SquareObject);

        NeuroSharp.Training.Network network = new NeuroSharp.Training.Network(14);
        network.Add(new NeuroSharp.FullyConnectedLayer(96));
        network.Add(new NeuroSharp.ActivationLayer(NeuroSharp.Enumerations.ActivationType.Tanh));
        network.Add(new NeuroSharp.FullyConnectedLayer(48));
        network.Add(new NeuroSharp.ActivationLayer(NeuroSharp.Enumerations.ActivationType.Tanh));
        network.Add(new NeuroSharp.FullyConnectedLayer(3));
        network.Add(new NeuroSharp.ActivationLayer(NeuroSharp.Enumerations.ActivationType.Tanh));
        network.UseLoss(NeuroSharp.Enumerations.LossType.MeanSquaredError);

        Learner = new ReinforcementLearner<Move>(network, NeuroSharp.Enumerations.TrainingConfiguration.SGD,
            NeuroSharp.Enumerations.OptimizerType.Adam, learningRate: 0.0004, 0.001f, 7);
    }

    void Update()
    {
        if (!SnakeGame.FoodExists)
            SnakeGame.SpawnFood();

        MoveTimer += Time.deltaTime * TimeStep;
        if(MoveTimer > GameTick)
        {
            MoveTimer = 0;

            List<Move> moves = new List<Move>();
            foreach(Move m in Enum.GetValues(typeof(Move)))
                moves.Add(m);

            FlattenedGameSpaceData = Snake.GetSnakeVision();
            Move nextMove = Learner.GetNextMove((int)LastMoveResult, CurrentMoveReward, 
                Score, moves, FlattenedGameSpaceData);

            SnakeMoveResult result = Snake.MoveSnake(nextMove);

            if (result.OutOfBounds || result.HitSelf || MovesWithoutEating > MovesWithoutEatingLimit)
            {
                CurrentMoveReward = NegativeMoveReward;
                Learner.TrainStep(5000);
                SnakeGame.ResetSnake();
                LastMoveResult = MoveResult.Death;
                MovesWithoutEating = 0;
                Score = 0;
            }
            else if (result.AteFood)
            {
                CurrentMoveReward = PositiveMoveReward;
                LastMoveResult = MoveResult.Eat;
                Learner.TrainStep(20);
                MovesWithoutEating = 0;
                Score++;
            }
            else
            {
                CurrentMoveReward = NeutralMoveReward;
                LastMoveResult = MoveResult.Nothing;
                Learner.TrainStep(1);
                MovesWithoutEating++;
            }

            if (result.AteFood)
                Debug.Log("Ate food");

            UpdateSnakeVisionVariables();
        }
    }

    void UpdateSnakeVisionVariables()
    {
        ObstacleLeft = (int)FlattenedGameSpaceData[0];
        ObstacleStraight = (int)FlattenedGameSpaceData[1];
        ObstacleRight = (int)FlattenedGameSpaceData[2];

        Up = (int)FlattenedGameSpaceData[3];
        Right = (int)FlattenedGameSpaceData[5];
        Down = (int)FlattenedGameSpaceData[4];
        Left = (int)FlattenedGameSpaceData[6];

        FoodUp = (int)FlattenedGameSpaceData[10];
        FoodRight = (int)FlattenedGameSpaceData[9];
        FoodDown = (int)FlattenedGameSpaceData[8];
        FoodLeft = (int)FlattenedGameSpaceData[7];

        FoodDirectlyLeft = (int)FlattenedGameSpaceData[11];
        FoodDirectlyStraight = (int)FlattenedGameSpaceData[12];
        FoodDirectlyRight = (int)FlattenedGameSpaceData[13];
    }
}
