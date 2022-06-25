using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public class GameManager : MonoBehaviour
{
    [Header("Initial Snake Parameters")]
    public int InitialSnakeLength;
    public float SnakeSegmentScale;

    public SpriteRenderer Segment;
    public SnakeAgent Snake;

    [Header("Test Movement")]
    public bool AddSegment;
    public bool CanMove;
    public Direction NextDirection;

    [Header("Food")]
    public GameObject FoodObj;
    public GameObject Food;

    [Header("Borders")]
    public GameObject LeftBorder;
    public GameObject RightBorder;
    public GameObject TopBorder;
    public GameObject BottomBorder;

    public GameState GameState;
    public Training Training;

    [Header("Training Variables")]
    public float DangerOffset;
    public float RandomMoveProbability;
    public int Batchsize;
    public int GameCount;

    [Header("Debug Variables")]
    public bool DebugMode;
    public GameObject LeftDanger;
    public GameObject RightDanger;
    public GameObject StraightDanger;
    public bool DangerLeft;
    public bool DangerRight;
    public bool DangerStraight;
    public AbsoluteDirection AbsoluteDirection;

    void Start()
    {
        Application.targetFrameRate = 120;
        CanMove = true;
        Snake = new SnakeAgent(InitialSnakeLength, SnakeSegmentScale, Segment, this);
        GameState = new GameState(this, DangerOffset, SnakeSegmentScale);
        Training = new Training(GameState, Batchsize);
        SpawnFood();
        if(DebugMode)
            SpawnDebugObjects();
    }

    void Update()
    {     
        if (CanMove)
        {
            RandomMoveProbability = Mathf.Max(0, 80 - GameCount);

            Vector<float> lastState = GameState.StateVector;
            if (Random.Range(0f, 100f) <= RandomMoveProbability)
                NextDirection = (Direction)Random.Range(0, 3);
            else
                NextDirection = Training.GetNextDirection(lastState);
            Snake.Move(NextDirection);
            //CanMove = false;

            //game over
            if (Snake.GameOver()) 
            {
                Training.UpdateNetwork(originState: lastState, directionTaken: NextDirection, rewardObtained: -10, trainShortMemory: false, diedThisTurn: true);
                StartNewGame();
                GameCount++;
            }
            //got food
            else if (Food.transform.position == Snake.Segments[0].transform.position)
            {
                Training.UpdateNetwork(originState: lastState, directionTaken: NextDirection, rewardObtained: 10, trainShortMemory: true, diedThisTurn: false);
                SpawnFood();
                Snake.AddSegment();
            }
            //nothing happened
            else
            {
                Training.UpdateNetwork(originState: lastState, directionTaken: NextDirection, rewardObtained: 0f, trainShortMemory: true, diedThisTurn: false);
            }
        }

        //debug only
        if (AddSegment)
        {
            Snake.AddSegment();
            AddSegment = false;
        }

        UpdateDebugObject();
    }

    void StartNewGame()
    {
        foreach(GameObject g in Snake.Segments)
            Destroy(g);
        Snake = new SnakeAgent(InitialSnakeLength, SnakeSegmentScale, Segment, this);
    }

    void SpawnFood()
    {
        if(Food != null)
            Destroy(Food);
        Vector3 position = new Vector3(Random.Range(LeftBorder.transform.position.x + 3*SnakeSegmentScale, RightBorder.transform.position.x - 3*SnakeSegmentScale), 
                                       Random.Range(BottomBorder.transform.position.y + 3 * SnakeSegmentScale, TopBorder.transform.position.y - 3*SnakeSegmentScale), 0);

        position.x = Mathf.Round(position.x / SnakeSegmentScale) * SnakeSegmentScale;
        position.y = Mathf.Round(position.y / SnakeSegmentScale) * SnakeSegmentScale;
        Food = GameObject.Instantiate(FoodObj, position, Quaternion.identity);
        Food.transform.localScale = Vector3.one * SnakeSegmentScale;
    }

    void SpawnDebugObjects()
    {
        //test objects for danger
        LeftDanger = new GameObject("Left Danger");
        RightDanger = new GameObject("Right Danger");
        StraightDanger = new GameObject("Straight Danger");

        LeftDanger.transform.localScale = Vector3.one * SnakeSegmentScale;
        RightDanger.transform.localScale = Vector3.one * SnakeSegmentScale;
        StraightDanger.transform.localScale = Vector3.one * SnakeSegmentScale;

        LeftDanger.AddComponent<SpriteRenderer>();
        RightDanger.AddComponent<SpriteRenderer>();
        StraightDanger.AddComponent<SpriteRenderer>();

        LeftDanger.GetComponent<SpriteRenderer>().sprite = Segment.sprite;
        LeftDanger.GetComponent<SpriteRenderer>().sharedMaterial = Segment.sharedMaterial;

        RightDanger.GetComponent<SpriteRenderer>().sprite = Segment.sprite;
        RightDanger.GetComponent<SpriteRenderer>().sharedMaterial = Segment.sharedMaterial;

        StraightDanger.GetComponent<SpriteRenderer>().sprite = Segment.sprite;
        StraightDanger.GetComponent<SpriteRenderer>().sharedMaterial = Segment.sharedMaterial;
    }

    void UpdateDebugObject()
    {
        if(DebugMode)
        {
            LeftDanger.transform.position = GameState.AbsoluteDirection == AbsoluteDirection.Left ? GameState.DownPoint :
                                            GameState.AbsoluteDirection == AbsoluteDirection.Right ? GameState.UpPoint :
                                            GameState.AbsoluteDirection == AbsoluteDirection.Up ? GameState.LeftPoint : GameState.RightPoint;

            RightDanger.transform.position = GameState.AbsoluteDirection == AbsoluteDirection.Left ? GameState.UpPoint :
                                             GameState.AbsoluteDirection == AbsoluteDirection.Right ? GameState.DownPoint :
                                             GameState.AbsoluteDirection == AbsoluteDirection.Up ? GameState.RightPoint : GameState.LeftPoint;

            StraightDanger.transform.position = GameState.AbsoluteDirection == AbsoluteDirection.Left ? GameState.LeftPoint :
                                                GameState.AbsoluteDirection == AbsoluteDirection.Right ? GameState.RightPoint :
                                                GameState.AbsoluteDirection == AbsoluteDirection.Up ? GameState.UpPoint : GameState.DownPoint;
        }

        DangerLeft = GameState.DangerLeft;
        DangerRight = GameState.DangerRight;
        DangerStraight = GameState.DangerStraight;
        AbsoluteDirection = GameState.AbsoluteDirection;
    }
}
