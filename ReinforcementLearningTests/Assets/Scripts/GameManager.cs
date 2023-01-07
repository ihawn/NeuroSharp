using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

public class GameManager : MonoBehaviour
{
    public SnakeGame SnakeGame;
    public GameObject SquareObject;
    public QLearner QLearner;
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

    Snake Snake { get { return SnakeGame.Snake; } }

    void Start()
    {
        SnakeGame = new SnakeGame(GameSizeX, GameSizeY, SnakeStartLength, SquareObject);
        QLearner = new QLearner(SnakeGame, 0.02f);
    }

    void Update()
    {
        MoveTimer += Time.deltaTime * TimeStep;
        if(MoveTimer > GameTick)
        {
            MoveTimer = 0;

            Vector<double> flattenedGameSpaceData = Snake.GetSnakeVision();
             /*   Vector<double>.Build.DenseOfEnumerable(
                    SnakeGame.GameSpaceMatrix.Cast<SnakeGameSquare>().ToList()
                    .Select(x => x.ContainsSnakeSegment ? 1d : 0d).ToList()
                );*/

           // Debug.Log(flattenedGameSpaceData[0] + ", " + flattenedGameSpaceData[1] + ", " + flattenedGameSpaceData[2]);

            Move nextMove = QLearner.GetNextMove(CurrentMoveReward, flattenedGameSpaceData);
            bool outOfBounds = Snake.MoveSnake(nextMove);

            if (outOfBounds)
            {
                CurrentMoveReward = NegativeMoveReward;
                SnakeGame.ResetSnake();
            }
            else
                CurrentMoveReward = NeutralMoveReward;
        }
    }
}
