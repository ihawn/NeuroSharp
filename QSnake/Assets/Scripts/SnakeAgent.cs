using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class SnakeAgent
{
    public List<GameObject> Segments { get; set; }
    public Vector3 AbsoluteDirectionVector
    {
        get { return (Segments[0].transform.position - Segments[1].transform.position).normalized; }
    }
    private GameManager gm { get; set; }
    private SpriteRenderer Renderer { get; set; }
    private float SegmentScale { get; set; }
    int MoveCount { get; set; }

    public SnakeAgent(int initialLength, float scale, SpriteRenderer r, GameManager gameManager)
    {
        gm = gameManager;
        Segments = new List<GameObject>();
        SegmentScale = scale;
        Renderer = r;
        for(int i = 0; i < initialLength; i++)
        {
            AddSegment();
        }
    }

    public void Move(Direction d)
    {
        MoveCount++;

        Vector3 direction = Segments[0].transform.position - Segments[1].transform.position;
        switch (d)
        {
            case Direction.Straight:
                //direction already correct
                break;
            case Direction.Left:
                direction = Quaternion.Euler(0, 0, 90) * direction;
                break;
            case Direction.Right:
                direction = Quaternion.Euler(0, 0, -90) * direction;
                break;
        }

        Vector3 nextPosition = Segments[0].transform.position;
        Vector3 lastPostion;
        for (int i = 1; i < Segments.Count; i++)
        {
            lastPostion = Segments[i].transform.position;
            Segments[i].transform.position = nextPosition;
            nextPosition = lastPostion;
        }

        Segments[0].transform.position += direction;
    }

    public void AddSegment()
    {
        GameObject g = new GameObject("segment " + (Segments.Count + 1));
        g.AddComponent<SpriteRenderer>();
        g.GetComponent<SpriteRenderer>().sprite = Renderer.sprite;
        g.GetComponent<SpriteRenderer>().sharedMaterial = Renderer.sharedMaterial;
        g.transform.localScale = Vector3.one * SegmentScale * 0.95f;
        if (Segments.Count == 0)
            g.transform.position = Vector3.zero;
        else if (Segments.Count == 1)
            g.transform.position = -new Vector3(SegmentScale, 0, 0) - Segments[0].transform.position;
        else
            g.transform.position = 2 * Segments[Segments.Count - 1].transform.position - Segments[Segments.Count - 2].transform.position;
        Segments.Add(g);
    }

    public bool GameOver()
    {
        return OutsideBorder(Segments[0].transform.position) || RanIntoSelf() || MoveCount > 100 * Segments.Count;
    }

    public bool IsCollision(Vector3 pt)
    {
        return OutsideBorder(pt) || RanIntoSelf(pt);
    }

    bool OutsideBorder(Vector3 pt)
    {
        return pt.x < gm.LeftBorder.transform.position.x + SegmentScale ||
               pt.x > gm.RightBorder.transform.position.x - SegmentScale ||
               pt.y < gm.BottomBorder.transform.position.y + SegmentScale ||
               pt.y > gm.TopBorder.transform.position.y - SegmentScale;
    }

    bool RanIntoSelf(Vector3? pt = null)
    {
        if (pt != null) //checking if non-segment point hits any segments
        {
            if (Segments.Any(s => Mathf.Abs(pt.Value.x - s.transform.position.x) <= 0.0001f &&
                                  Mathf.Abs(pt.Value.y - s.transform.position.y) <= 0.0001f))
            {
                return true;
            }
        }
        else //checking if segment point (on the snake) hits another segment
        {
            foreach ((float, float) xy in Segments.Select(s => (s.transform.position.x, s.transform.position.y)))
            {
                if (Segments.Where(s => Mathf.Abs(s.transform.position.x - xy.Item1) < 0.0001f && Mathf.Abs(s.transform.position.y - xy.Item2) < 0.0001f).Count() > 1)
                {
                    return true;
                }
            }
        }
        return false;
    }
}

public enum Direction
{
    Straight = 0,
    Left = 1,
    Right = 2
}

public enum AbsoluteDirection
{
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3
}