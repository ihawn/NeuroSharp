using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using UnityEngine;
using UnityEditor;

[assembly: InternalsVisibleTo("Unity.Burst.Editor.Tests")]

namespace Unity.Burst.Editor
{
    internal class LongTextArea
    {
        private const float naturalEnhancedPad = 20f;
        private const int kMaxFragment = 2048;

        private struct Fragment
        {
            public int lineCount;
            public string text;
        }

        internal float fontHeight = 0.0f;
        internal float fontWidth = 0.0f;

        private string m_Text = "";
        private int _mTextLines = 0;
        private List<Fragment> m_Fragments = null;
        private bool invalidated = true;
        internal Vector2 finalAreaSize;

        private static readonly Texture2D backgroundTexture = Texture2D.whiteTexture;
        private static readonly GUIStyle textureStyle = new GUIStyle { normal = new GUIStyleState { background = backgroundTexture } };

        internal float horizontalPad = 50.0f;

        private int[] _lineDepth = null;
        private bool[] _folded = null;
        internal int[] _blockLine = null;
        private List<Fragment>[] _blocksFragments = null;
        private readonly string _foldedString = "...\n";

        private BurstDisassembler _disassembler;

        private int _selectBlockStart = -1;
        private float _selectStartY = -1f;
        private int _selectBlockEnd = -1;
        private float _selectEndY = -1f;

        private float _renderStartY = 0.0f;
        private int _renderBlockStart = -1;
        private int _renderBlockEnd = -1;
        private int _initialLineCount = -1;

        private bool _mouseDown = false;
        private bool _mouseOutsideBounds = true;
        internal Vector2 _selectPos = Vector2.zero;
        internal Vector2 _selectDragPos = Vector2.zero;

        private Color _selectionColor;
        private Color _selectionColorDarkmode = new Color(0f, .6f, .9f, .5f);
        private Color _selectionColorLightmode = new Color(0f, 0f, .9f, .2f);

        private readonly Color[] _colourWheel = new Color[]
               {Color.blue, Color.cyan, Color.green, Color.magenta, Color.red, Color.yellow, Color.white, Color.black};

        internal (int idx, int length) _textSelectionIdx = (0, 0);
        private bool _textSelectionIdxValid = true;

        internal (int blockIdx, int textIdx) _enhancedTextSelectionIdxStart = (0, 0);
        internal (int blockIdx, int textIdx) _enhancedTextSelectionIdxEnd = (0, 0);

        private bool _oldShowBranchMarkers = false;

        internal const int regLineThickness = 2;
        private const int highlightLineThickness = 3;
        private float _verticalPad = 0;

        internal Branch _hoveredBranch;
        private BurstDisassembler.AsmEdge _prevHoveredEdge;

        private float _hitBoxAdjust = 2f;

        internal bool IsTextSet(string textToRender)
        {
            return textToRender == m_Text;
        }

        public void SetText(string textToRender, bool isDarkMode, bool isAssemblyKindJustChanged, BurstDisassembler disassembler, bool useDisassembler)
        {
            StopSelection(); // we just switched view
            m_Text = textToRender;
            if (!useDisassembler)
            {
                _disassembler = null;
                m_Fragments = RecomputeFragments(m_Text);
                horizontalPad = 0.0f;
            }
            else
            {
                m_Fragments = null;
                SetDisassembler(disassembler);
                _selectionColor = isDarkMode ? _selectionColorDarkmode : _selectionColorLightmode;
            }

            invalidated = true;
        }

        public void ExpandAllBlocks()
        {
            StopSelection();
            int blockIdx = 0;
            foreach (var block in _disassembler.Blocks)
            {
                var changed = _folded[blockIdx];
                _folded[blockIdx] = false;
                if (changed)
                {
                    finalAreaSize.y += Math.Max(block.Length - 1, 1) * fontHeight;
                }
                blockIdx++;
            }
        }

        public void FocusCodeBlocks()
        {
            StopSelection();
            var blockIdx = 0;
            foreach (var block in _disassembler.Blocks)
            {
                bool changed = false;
                switch (block.Kind)
                {
                    case BurstDisassembler.AsmBlockKind.None:
                    case BurstDisassembler.AsmBlockKind.Directive:
                    case BurstDisassembler.AsmBlockKind.Block:
                    case BurstDisassembler.AsmBlockKind.Data:
                        if (!_folded[blockIdx])
                        {
                            changed = true;
                        }
                        _folded[blockIdx] = true;
                        break;
                    case BurstDisassembler.AsmBlockKind.Code:
                        if (_folded[blockIdx])
                        {
                            changed = true;
                        }
                        _folded[blockIdx] = false;
                        break;
                }

                if (changed)
                {
                    if (_folded[blockIdx])
                    {
                        finalAreaSize.y -= Math.Max(block.Length - 1, 1) * fontHeight;
                    }
                    else
                    {
                        finalAreaSize.y += Math.Max(block.Length - 1, 1) * fontHeight;
                    }

                }
                blockIdx++;
            }
        }

        private void ComputeInitialLineCount()
        {
            var blockIdx = 0;
            _initialLineCount = 0;
            foreach (var block in _disassembler.Blocks)
            {
                switch (block.Kind)
                {
                    case BurstDisassembler.AsmBlockKind.None:
                    case BurstDisassembler.AsmBlockKind.Directive:
                    case BurstDisassembler.AsmBlockKind.Block:
                    case BurstDisassembler.AsmBlockKind.Data:
                        _folded[blockIdx] = true;
                        break;
                    case BurstDisassembler.AsmBlockKind.Code:
                        _folded[blockIdx] = false;
                        break;
                }
                _initialLineCount += _folded[blockIdx] ? 1 : block.Length;
                blockIdx++;
            }
        }

        public void SetDisassembler(BurstDisassembler disassembler)
        {
            _disassembler = disassembler;
            if (disassembler == null)
            {
                return;
            }

            var numBlocks = _disassembler.Blocks.Count;
            var numLinesFromBlock = new int[numBlocks];
            _lineDepth = new int[numBlocks];
            _folded = new bool[numBlocks];
            _blockLine = new int[numBlocks];
            _blocksFragments = new List<Fragment>[numBlocks];

            ComputeInitialLineCount();

            // Count edges
            var edgeCount = 0;
            foreach (var block in _disassembler.Blocks)
            {
                if (block.Edges != null)
                {
                    foreach (var edge in block.Edges)
                    {
                        if (edge.Kind == BurstDisassembler.AsmEdgeKind.OutBound)
                        {
                            edgeCount++;
                        }
                    }
                }
            }

            var edgeArray = new BurstDisassembler.AsmEdge[edgeCount];
            var edgeIndex = 0;
            foreach (var block in _disassembler.Blocks)
            {
                if (block.Edges != null)
                {
                    foreach (var edge in block.Edges)
                    {
                        if (edge.Kind == BurstDisassembler.AsmEdgeKind.OutBound)
                        {
                            edgeArray[edgeIndex++] = edge;
                        }
                    }
                }
            }

            Array.Sort(edgeArray, (a, b) =>
            {
                var src1BlockIdx = a.OriginRef.BlockIndex;
                var src1Line = _disassembler.Blocks[src1BlockIdx].LineIndex;
                src1Line += a.OriginRef.LineIndex;
                var dst1BlockIdx = a.LineRef.BlockIndex;
                var dst1Line = _disassembler.Blocks[dst1BlockIdx].LineIndex;
                dst1Line += a.LineRef.LineIndex;
                var Len1 = Math.Abs(src1Line - dst1Line);
                var src2BlockIdx = b.OriginRef.BlockIndex;
                var src2Line = _disassembler.Blocks[src2BlockIdx].LineIndex;
                src2Line += b.OriginRef.LineIndex;
                var dst2BlockIdx = b.LineRef.BlockIndex;
                var dst2Line = _disassembler.Blocks[dst2BlockIdx].LineIndex;
                dst2Line += b.LineRef.LineIndex;
                var Len2 = Math.Abs(src2Line - dst2Line);
                return Len1 - Len2;
            });

            // Iterate through the blocks to pre-compute the widths for branches
            int maxLine = 0;
            foreach (var edge in edgeArray)
            {
                if (edge.Kind == BurstDisassembler.AsmEdgeKind.OutBound)
                {
                    var s = edge.OriginRef.BlockIndex;
                    var e = edge.LineRef.BlockIndex;
                    if (e == s + 1)
                    {
                        continue;   // don't render if its pointing to next line
                    }

                    int m = 0;

                    int l = s;
                    int le = e;
                    if (e < s)
                    {
                        l = e;
                        le = s;
                    }

                    for (; l <= le; l++)
                    {
                        numLinesFromBlock[l]++;
                        if (m < numLinesFromBlock[l])
                        {
                            m = numLinesFromBlock[l];
                        }
                        if (maxLine < m)
                        {
                            maxLine = m;
                        }
                    }

                    _lineDepth[s] = m;
                }
            }

            horizontalPad = naturalEnhancedPad + maxLine * 10;
        }

        // Changing the font size doesn't update the text field, so added this to force a recalculation
        public void Invalidate()
        {
            invalidated = true;
        }

        private struct HoverBox
        {
            public Vector2 position;
            public string info;
            public bool valid;
        }

        private HoverBox hover;

        public void Interact(Vector2 pos)
        {
            if (_disassembler == null)
            {
                return;
            }

            // lineNumber (absolute)
            int lineNumber = Mathf.FloorToInt(pos.y / fontHeight);
            int blockNumber = 0;
            for (int idx = 0; idx < _disassembler.Blocks.Count; idx++)
            {
                var block = _disassembler.Blocks[idx];
                if (lineNumber < block.LineIndex)
                {
                    break;
                }
                if (_folded[idx])
                {
                    lineNumber += block.Length - 1;
                }
            }
            int column = Mathf.FloorToInt((pos.x - horizontalPad) / fontWidth);

            hover.valid = false;

            if (column < 0 || lineNumber < 0 || lineNumber >= _disassembler.Lines.Count)
            {
                return;
            }

            int tokIdx;
            try
            {
                tokIdx = _disassembler.GetTokenIndexFromColumn(blockNumber, lineNumber, column, out _);
            }
            catch (ArgumentOutOfRangeException)
            {
                return;
            }
            if (tokIdx < 0)
            {
                return;
            }

            var tok = _disassembler.GetToken(tokIdx);

            // Found match
            hover.valid = true;
            hover.position = pos;
            hover.info = $"Token Hit : {tok.Kind} - {tok.Position} - {tok.Length}";
        }

        // Not clipped, so use only if you know the line will be within the ui element
        private void DrawLine(Vector2 start, Vector2 end, int width)
        {
            var matrix = GUI.matrix;
            Vector2 distance = end - start;
            float angle = Mathf.Rad2Deg * Mathf.Atan(distance.y / distance.x);
            if (distance.x < 0)
            {
                angle += 180;
            }

            int width2 = (int)Mathf.Ceil(width / 2);

            Rect pos = new Rect(start.x, start.y - width2, distance.magnitude, width);

            GUIUtility.RotateAroundPivot(angle, start);
            GUI.DrawTexture(pos, backgroundTexture);
            GUI.matrix = matrix; // restore initial matrix to avoid floating point inaccuracies with RotateAroundPivot(...)
        }

        private void DrawLine(Rect line, float angle)
        {
            var matrix = GUI.matrix;
            GUIUtility.RotateAroundPivot(angle, /*new Vector2(line.xMin, line.center.y)*/ /*start*/ new Vector2(line.x, line.center.y));
            GUI.DrawTexture(line, backgroundTexture);
            GUI.matrix = matrix; // restore initial matrix to avoid floating point inaccuracies with RotateAroundPivot(...)
        }

        private void DrawHover(GUIStyle style)
        {
            if (hover.valid)
            {
                GUI.Box(new Rect(horizontalPad + hover.position.x, hover.position.y, (hover.info.Length + 1) * fontWidth,
                    2 * fontHeight), "", textureStyle);
                GUI.color = Color.black;
                GUI.Label(new Rect(horizontalPad + hover.position.x + fontWidth * 0.5f, hover.position.y + fontHeight * 0.5f, hover.info.Length * fontWidth, fontHeight), hover.info, style);
            }
        }

        private Vector2 AnglePoint(float angle, Vector2 point, Vector2 pivotPoint)
        {
            // https://matthew-brett.github.io/teaching/rotation_2d.html
            // Problem Angle is calculates as angle clockwise, and here we use it as it was counterclockwise!
            var s = Mathf.Sin(angle);
            var c = Mathf.Cos(angle);
            point -= pivotPoint;

            // This is wrong, as we need not rotate around origo, but the rect!
            return new Vector2(c * point.x - s * point.y, s * point.x + c * point.y) + pivotPoint;
        }

        private float CalculateAngle(Vector2 start, Vector2 end)
        {
            Vector2 distance = end - start;
            float angle = Mathf.Rad2Deg * Mathf.Atan(distance.y / distance.x);
            if (distance.x < 0)
            {
                angle += 180;
            }

            return angle;
        }

        internal struct Branch
        {
            public BurstDisassembler.AsmEdge Edge;

            public Rect StartHorizontal;
            public Rect VerticalLine;
            public Rect EndHorizontal;

            public Rect UpperLine;
            public Rect LowerLine;
            public float UpperAngle;
            public float LowerAngle;

            public Branch(BurstDisassembler.AsmEdge edge, Rect startHorizontal, Rect verticalLine, Rect endHorizontal, Rect upperLine, Rect lowerLine, float angle1, float angle2)
            {
                Edge = edge;

                StartHorizontal = startHorizontal;
                VerticalLine = verticalLine;
                EndHorizontal = endHorizontal;

                UpperLine = upperLine;
                LowerLine = lowerLine;
                UpperAngle = angle1;
                LowerAngle = angle2;
            }
        }

        private void MakeBranchThin(ref Branch branch)
        {
            int lineThickness = regLineThickness;

            branch.StartHorizontal.height = lineThickness;
            branch.VerticalLine.width = lineThickness;
            branch.EndHorizontal.height = lineThickness;

            branch.UpperLine.height = lineThickness;
            branch.LowerLine.height = lineThickness;

            // Adjusting position for arrowtip for thicker lines.
            branch.UpperLine.y += (highlightLineThickness - regLineThickness);

            branch.UpperLine.position -= new Vector2(.5f, .5f);
            branch.LowerLine.position -= new Vector2(.5f, -.5f);

            // Make end part of arrow expand upwards.
            branch.EndHorizontal.y += (highlightLineThickness - regLineThickness);
        }

        /// <summary>
        /// Use this for hover, as there is a slight visual misbalance
        /// between cursor position and visual cursor.
        /// </summary>
        private Vector2 GetMousePosForHover()
        {
            Vector2 mousePos = Event.current.mousePosition;
            mousePos.y -= 0.5f;
            mousePos.x += 0.5f;
            return mousePos;
        }

        /// <summary>
        /// Calculate the position and size of an edge, and return it as a branch.
        /// </summary>
        /// <param name="edge"> The edge to base branch on. </param>
        /// <param name="x"> Start x position of branch. </param>
        /// <param name="y1"> Start y position of branch. </param>
        /// <param name="y2"> End y position of branch. </param>
        /// <param name="w"> Depth of branch. </param>
        private Branch CalculateBranch(BurstDisassembler.AsmEdge edge, float x, float y1, float y2, int w)
        {
            bool isEdgeHovered = edge.Equals(_prevHoveredEdge);

            int lineThickness = isEdgeHovered
                ? highlightLineThickness
                : regLineThickness;

            // Calculate rectangles for branch arrows:
            var start = new Vector2(x, y1 + _verticalPad);
            var end = start + new Vector2(-(w * 10), 0);

            var branchStartPos = new Rect(end.x - lineThickness / 2, start.y - 1, start.x - end.x + lineThickness / 2, lineThickness);

            start = end;
            end = start + new Vector2(0, y2 - y1);

            var branchVerticalPartPos = end.y < start.y
                ? new Rect(start.x - 1, end.y, lineThickness, start.y - end.y)
                : new Rect(start.x - 1, start.y, lineThickness, end.y - start.y);

            start = end;
            end = start + new Vector2(w * 10, 0);

            var branchEndPos = new Rect(start.x - lineThickness / 2, start.y - 1, end.x - start.x, lineThickness);

            // Calculate rectangels for arrowtip.
            Vector2 lowerArrowTipStart = end;
            Vector2 upperArrowtipStart = end;

            //   Moving the arrowtips closer together.
            upperArrowtipStart += new Vector2(0, 0.5f);
            lowerArrowTipStart -= new Vector2(0, 0.5f);

            //   Upper arrowtip.
            var upperArrowTipEnd = upperArrowtipStart + new Vector2(-5, -5);

            var upperLine = new Rect(upperArrowtipStart.x, upperArrowtipStart.y - (int)Mathf.Ceil(lineThickness / 2), (upperArrowTipEnd - upperArrowtipStart).magnitude, lineThickness);

            //   Lower arrowtip.
            var lowerArrowtipEnd = lowerArrowTipStart + new Vector2(-5, 5);
            var lowerLine = new Rect(lowerArrowTipStart.x, lowerArrowTipStart.y - (int)Mathf.Ceil(lineThickness / 2), (lowerArrowtipEnd - lowerArrowTipStart).magnitude, lineThickness);

            if (isEdgeHovered)
            {
                // Adjusting position for arrowtip for thicker lines.
                upperArrowtipStart.y -= (highlightLineThickness - regLineThickness);
                upperArrowTipEnd.y -= (highlightLineThickness - regLineThickness);
                upperLine.y -= (highlightLineThickness - regLineThickness);

                upperLine.position += new Vector2(.5f, .5f);
                lowerLine.position += new Vector2(.5f, -.5f);

                // Make end part of arrow expand upwards.
                branchEndPos.y -= (highlightLineThickness - regLineThickness);
            }

            var branch = new Branch(edge, branchStartPos, branchVerticalPartPos, branchEndPos, upperLine, lowerLine,
                CalculateAngle(upperArrowtipStart, upperArrowTipEnd), CalculateAngle(lowerArrowTipStart, lowerArrowtipEnd));

            // Handling wether mouse is hovering over edge.
            Vector2 mousePos = GetMousePosForHover();

            // Rotate mousePos so it seems like lower arrow tip is not rotatet.
            Vector2 lowerArrowTipPivot = lowerArrowTipStart;
            lowerArrowTipPivot.y -= (int)Mathf.Ceil(lineThickness / 2);
            Vector2 angledMouseLower = AnglePoint(CalculateAngle(lowerArrowTipPivot, lowerArrowtipEnd), mousePos, new Vector2(lowerLine.x, lowerLine.center.y));
            angledMouseLower.y -= (int)Mathf.Ceil(lineThickness / 2);

            Vector2 upperArrowTipPivot = upperArrowtipStart;
            upperArrowTipPivot.y += (int)Mathf.Ceil(lineThickness / 2);
            Vector2 angleMouseUpper = AnglePoint(CalculateAngle(upperArrowTipPivot, upperArrowTipEnd) - 360, mousePos, new Vector2(upperLine.x, upperLine.center.y));
            angleMouseUpper.y += (int)Mathf.Ceil(lineThickness / 2);

            if (AdjustedContains(branchStartPos, mousePos) || AdjustedContains(branchVerticalPartPos, mousePos) || AdjustedContains(branchEndPos, mousePos)
                || AdjustedContains(lowerLine, angledMouseLower) || AdjustedContains(upperLine, angleMouseUpper))
            {
                // Handling whether another branch was already made thick is done in DrawBranch(...).
                if (!_hoveredBranch.Edge.Equals(default(BurstDisassembler.AsmEdge)) && _hoveredBranch.Edge.Equals(_prevHoveredEdge))
                {
                    return branch;
                }
                _hoveredBranch = branch;
            }
            return branch;
        }

        private bool AdjustedContains(Rect rect, Vector2 point)
        {
            return rect.yMax + _hitBoxAdjust >= point.y && rect.yMin - _hitBoxAdjust <= point.y
                    && rect.xMax + _hitBoxAdjust >= point.x && rect.xMin - _hitBoxAdjust <= point.x;
        }

        /// <summary>
        /// Draws a branch as a jumpable set of boxes.
        /// </summary>
        /// <param name="branch"> The branch to draw. </param>
        /// <param name="w"> Depth of the branch. </param>
        /// <param name="colourWheel"> Array of possible colours for branches. </param>
        /// <param name="workingArea"> Current view in inspector. </param>
        private void DrawBranch(Branch branch, int w, Rect workingArea)
        {
            bool isBranchHovered = branch.Edge.Equals(_hoveredBranch.Edge);
            Vector2 scrollPos = workingArea.position;

            int lineThickness = isBranchHovered
                ? highlightLineThickness
                : regLineThickness;

            GUI.color = _colourWheel[w % _colourWheel.Length];

            // Check if hovered but not made thick yet:
            if (isBranchHovered && !branch.Edge.Equals(_prevHoveredEdge))
            {
                // alter thickness as edge is hovered over.
                branch.StartHorizontal.height = highlightLineThickness;
                branch.VerticalLine.width = highlightLineThickness;
                branch.EndHorizontal.height = highlightLineThickness;

                branch.UpperLine.height = highlightLineThickness;
                branch.LowerLine.height = highlightLineThickness;

                // Adjusting position for arrowtip for thicker lines.
                branch.UpperLine.y -= (highlightLineThickness - regLineThickness);

                branch.UpperLine.position += new Vector2(.5f, .5f);
                branch.LowerLine.position += new Vector2(.5f, -.5f);

                // Make end part of arrow expand upwards.
                branch.EndHorizontal.y -= (highlightLineThickness - regLineThickness);
            }
			else if (branch.EndHorizontal.height == highlightLineThickness && !isBranchHovered)
            {
                // the branch was previousy hovered, but is now hidden behind
                // another branch, that is the one being visually hovered.
                MakeBranchThin(ref branch);
            }

            // Render actual arrows:
            GUI.Box(branch.StartHorizontal, "", textureStyle);
            GUI.Box(branch.VerticalLine, "", textureStyle);

            float yy = branch.EndHorizontal.y - scrollPos.y;
            if (yy >= 0 && yy < workingArea.height)
            {
                GUI.Box(branch.EndHorizontal, "", textureStyle);
                DrawLine(branch.UpperLine, branch.UpperAngle);
                DrawLine(branch.LowerLine, branch.LowerAngle);
            }

            // Use below instead of buttons, to make the currently hovered edge the jumpable,
            // and not the edge rendered first i.e. when two edges overlap.
            if (Event.current.type == EventType.MouseDown && isBranchHovered)
            {
                Vector2 mousePos = GetMousePosForHover();

                // Rotate mousePos so it seems like lower arrow tip is not rotatet.
                Vector2 lowerLineEnd = branch.LowerLine.position;
                lowerLineEnd.y += (int)Mathf.Ceil(lineThickness / 2);
                lowerLineEnd += new Vector2(-5, 5);

                Vector2 angledMouseLower = AnglePoint(CalculateAngle(branch.LowerLine.position, lowerLineEnd),
                    mousePos, new Vector2(branch.LowerLine.x, branch.LowerLine.center.y));
                angledMouseLower.y -= (int)Mathf.Ceil(lineThickness / 2);

                // Rotate mousePos so it seems like upper arrow tip id not rotatet.
                Vector2 upperArrowTipPivot = branch.UpperLine.position;
                upperArrowTipPivot.y += 2 * (int)Mathf.Ceil(lineThickness / 2);

                Vector2 upperLineEnd = branch.UpperLine.position;
                upperLineEnd.y += (int)Mathf.Ceil(lineThickness / 2);
                upperLineEnd += new Vector2(-5, -5);

                Vector2 angleMouseUpper = AnglePoint(CalculateAngle(upperArrowTipPivot, upperLineEnd) - 360,
                    Event.current.mousePosition, new Vector2(branch.UpperLine.x, branch.UpperLine.center.y));
                angleMouseUpper.y += (int)Mathf.Ceil(lineThickness / 2);

                // Se if a jump should be made and jump.
                if (AdjustedContains(branch.StartHorizontal, mousePos))
                {
                    // make endarrow be at mouse cursor.
                    var target = branch.EndHorizontal;
                    target.y += branch.StartHorizontal.y < branch.EndHorizontal.y
                        ? (workingArea.yMax - mousePos.y)
                        : (workingArea.yMin - mousePos.y + highlightLineThickness / 2f);
                    GUI.ScrollTo(target);
                    Event.current.Use();
                }
                else if (AdjustedContains(branch.EndHorizontal, mousePos) || AdjustedContains(branch.LowerLine, angledMouseLower)
                    || AdjustedContains(branch.UpperLine, angleMouseUpper) || AdjustedContains(branch.VerticalLine, mousePos))
                {
                    var target = branch.StartHorizontal;
                    target.y += branch.StartHorizontal.y < branch.EndHorizontal.y
                        ? workingArea.yMin - mousePos.y + highlightLineThickness / 2
                        : workingArea.yMax - mousePos.y;
                    GUI.ScrollTo(target);
                    Event.current.Use();
                }
            }
        }

        private bool DrawFold(float x, float y, bool state, BurstDisassembler.AsmBlockKind kind)
        {
            var currentBg = GUI.backgroundColor;
            switch (kind)
            {
                case BurstDisassembler.AsmBlockKind.None:
                case BurstDisassembler.AsmBlockKind.Directive:
                case BurstDisassembler.AsmBlockKind.Block:
                    GUI.backgroundColor = Color.grey;
                    break;
                case BurstDisassembler.AsmBlockKind.Code:
                    GUI.backgroundColor = Color.green;
                    break;
                case BurstDisassembler.AsmBlockKind.Data:
                    GUI.backgroundColor = Color.magenta;
                    break;
            }

            var pressed = false;
            if (state)
            {
                pressed = GUI.Button(new Rect(x - fontWidth, y, fontWidth, fontHeight), "+");
            }
            else
            {
                pressed = GUI.Button(new Rect(x - fontWidth, y, fontWidth, fontHeight), "-");
            }

            GUI.backgroundColor = currentBg;

            return pressed;
        }

        private void Layout(GUIStyle style, float hPad)
        {
            var cacheSize0 = style.CalcSize(new GUIContent("W\n"));
            var cacheSize1 = style.CalcSize(new GUIContent("WW\n\n"));

            var oldFontHeight = fontHeight;
            var oldFontWidth = fontWidth;

            fontHeight = cacheSize1.y - cacheSize0.y;
            fontWidth = cacheSize1.x - cacheSize0.x;

            _verticalPad = fontHeight * 0.5f;

            // oldFontWidth == 0 means we picked the first target after opening inspector.
            var diffX = oldFontWidth != 0 ? fontWidth / oldFontWidth : 0.0f;

            if (HasSelection() && (oldFontWidth != fontWidth || oldFontHeight != fontHeight))
            {
                float diffY = fontHeight / oldFontHeight;

                // We only have to take padding into account for x-axis, as it's the only one with it.
                _selectPos = new Vector2(((_selectPos.x - hPad) * diffX) + hPad, diffY * _selectPos.y);
                _selectDragPos = new Vector2(((_selectDragPos.x - hPad) * diffX) + hPad, diffY * _selectDragPos.y);
            }

            invalidated = false;
            var oldFinalAreaSizeX = finalAreaSize.x;
            finalAreaSize = new Vector2(0.0f, 0.0f);

            if (_disassembler == null)
            {
                LayoutPlain(style);
            }
            else
            {
                finalAreaSize.y = _initialLineCount * fontHeight;
                finalAreaSize.x = oldFinalAreaSizeX * diffX;
            }
        }

        private void LayoutPlain(GUIStyle style)
        {
            // Using plain old text
            foreach (var frag in m_Fragments)
            {
                // Calculate the size as we have hidden control codes in the string
                var size = style.CalcSize(new GUIContent(frag.text));
                finalAreaSize.x = Math.Max(finalAreaSize.x, size.x);
                finalAreaSize.y += frag.lineCount * fontHeight;
            }
        }

        private void LayoutEnhanced(GUIStyle style, Rect workingArea, bool showBranchMarkers)
        {
            Vector2 scrollPos = workingArea.position;
            if (HasSelection() && showBranchMarkers != _oldShowBranchMarkers)
            {
                // need to alter selection according to padding on x-axis.
                if (showBranchMarkers)
                {
                    _selectPos.x += (horizontalPad - naturalEnhancedPad);
                    _selectDragPos.x += (horizontalPad - naturalEnhancedPad);
                }
                else
                {
                    _selectPos.x -= (horizontalPad - naturalEnhancedPad);
                    _selectDragPos.x -= (horizontalPad - naturalEnhancedPad);
                }
            }
            _oldShowBranchMarkers = showBranchMarkers;

            // Also computes the first and last blocks to render this time and ensures
            float positionY = 0.0f;
            int lNum = 0;
            _renderBlockStart = -1;
            _renderBlockEnd = -1;

            _selectBlockStart = -1;
            _selectStartY = -1f;
            _selectBlockEnd = -1;
            _selectEndY = -1f;

            for (int idx = 0; idx<_disassembler.Blocks.Count; idx++)
            {
                var block = _disassembler.Blocks[idx];
                var blockHeight = block.Length * fontHeight;
                var lHeight = block.Length;

                if (_folded[idx])
                {
                    blockHeight = fontHeight;
                    lHeight = 1;
                }

                _blockLine[idx] = lNum;

                if (_selectBlockStart == -1 && _selectPos.y - blockHeight <= positionY)
                {
                    _selectBlockStart = idx;
                    _selectStartY = positionY;
                }
                if (_selectBlockEnd == -1 && (_selectDragPos.y - blockHeight <= positionY || idx == _disassembler.Blocks.Count - 1))
                {
                    _selectBlockEnd = idx;
                    _selectEndY = positionY;
                }

                // Whole block is above view, or block starts below view.
                // If at last block and _renderBlockStart == -1, we must have had all block above our scrollPos.
                // Happens when Scroll view is at bottom and fontsize is decreased. As this forces the scrollview upwards,
                // we simply set the last block as the one being rendered (avoids an indexOutOfBounds exception).
                if (((positionY - scrollPos.y + blockHeight) < 0 || (positionY - scrollPos.y) > workingArea.size.y)
                    && !(idx == _disassembler.Blocks.Count - 1 && _renderBlockStart == -1))
                {
                    positionY += blockHeight;
                    lNum += lHeight;
                    continue;
                }

                if (_renderBlockStart == -1)
                {
                    _renderBlockStart = idx;
                    _renderStartY = positionY;
                }

                _renderBlockEnd = idx;

                if (_blocksFragments[idx] == null)
                {
                    _blocksFragments[idx] = RecomputeFragmentsFromBlock(idx);
                    foreach (var fragment in _blocksFragments[idx])
                    {
                        style.CalcMinMaxWidth(new GUIContent(fragment.text), out _, out var maxWidth);
                        if (finalAreaSize.x < maxWidth)
                        {
                            finalAreaSize.x = maxWidth;
                        }
                    }
                }

                positionY += blockHeight;
                lNum += lHeight;
            }
        }

        internal string GetStartingColorTag(string text)
        {
            if (_disassembler?.IsColored ?? false)
            {
                const int colorTagLength = 15;
                int colorTagIdx = text.LastIndexOf("<color=", _enhancedTextSelectionIdxStart.textIdx);

                // Checking that found tag is actually for this idx.
                if (colorTagIdx == -1)
                {
                    return "";
                }
                else
                {
                    return text.IndexOf("</color>", colorTagIdx, _enhancedTextSelectionIdxStart.textIdx - colorTagIdx) == -1
                        ? text.Substring(colorTagIdx, colorTagLength)
                        : "";
                }
            }
            else
            {
                return "";
            }
        }

        internal string GetEndingColorTag(string text)
        {
            if (_disassembler?.IsColored ?? false)
            {
                int endColorTagIdx = text.IndexOf("</color>", _enhancedTextSelectionIdxEnd.textIdx);

                // Check that tag actually belongs for this idx.
                if (endColorTagIdx == -1)
                {
                    return "";
                }
                else
                {
                    return text.IndexOf("<color=", _enhancedTextSelectionIdxEnd.textIdx, endColorTagIdx - _enhancedTextSelectionIdxEnd.textIdx) == -1
                        ? "</color>"
                        : "";
                }
            }
            else
            {
                return "";
            }
        }

        internal void DoSelectionCopy()
        {
            if (HasSelection())
            {
                if (_disassembler != null && _enhancedTextSelectionIdxEnd != _enhancedTextSelectionIdxStart)
                {
                    var str = new StringBuilder();

                    if (_enhancedTextSelectionIdxStart.blockIdx < _enhancedTextSelectionIdxEnd.blockIdx)
                    {
                        // Multiple blocks to copy from.
                        int blockIdx = _enhancedTextSelectionIdxStart.blockIdx;

                        if (_folded[blockIdx])
                        {
                            str.Append(_foldedString.Substring(_enhancedTextSelectionIdxStart.textIdx));
                        }
                        else
                        {
                            string text = _disassembler.GetOrRenderBlockToText(blockIdx);

                            str.Append(GetStartingColorTag(text) + text.Substring(_enhancedTextSelectionIdxStart.textIdx));
                        }
                        blockIdx++;

                        for (; blockIdx < _enhancedTextSelectionIdxEnd.blockIdx; blockIdx++)
                        {
                            if (_folded[blockIdx])
                            {
                                str.Append(_foldedString);
                            }
                            else
                            {
                                str.Append(_disassembler.GetOrRenderBlockToText(blockIdx));
                            }
                        }

                        if (_folded[blockIdx])
                        {
                            str.Append(_foldedString.Substring(0, _enhancedTextSelectionIdxEnd.textIdx));
                        }
                        else
                        {
                            string text = _disassembler.GetOrRenderBlockToText(blockIdx);

                            str.Append(text.Substring(0, _enhancedTextSelectionIdxEnd.textIdx) + GetEndingColorTag(text));
                        }

                    }
                    else
                    {
                        // Single block to copy from.
                        if (_folded[_enhancedTextSelectionIdxStart.blockIdx])
                        {
                            str.Append(_foldedString.Substring(_enhancedTextSelectionIdxStart.textIdx, _enhancedTextSelectionIdxEnd.textIdx - _enhancedTextSelectionIdxStart.textIdx));
                        }
                        else
                        {
                            string text = _disassembler.GetOrRenderBlockToText(_enhancedTextSelectionIdxStart.blockIdx);


                            str.Append(GetStartingColorTag(text)
                                + text.Substring(_enhancedTextSelectionIdxStart.textIdx, _enhancedTextSelectionIdxEnd.textIdx - _enhancedTextSelectionIdxStart.textIdx)
                                + GetEndingColorTag(text));
                        }
                    }

                    EditorGUIUtility.systemCopyBuffer = str.ToString();
                }
                else if (_textSelectionIdx.length > 0)
                {
                    EditorGUIUtility.systemCopyBuffer = m_Text.Substring(_textSelectionIdx.idx, _textSelectionIdx.length);
                }
            }
        }

        internal void SelectAll()
        {
            _selectPos = Vector2.zero;
            _selectDragPos = finalAreaSize;

            if (_disassembler != null)
            {
                _enhancedTextSelectionIdxStart = (0, 0);
                int blockIdx = _disassembler.Blocks.Count - 1;
                _enhancedTextSelectionIdxEnd = (blockIdx,
                    !_folded[blockIdx] ? _disassembler.GetOrRenderBlockToText(blockIdx).Length - 1 : _foldedString.Length - 1);
            }
            else
            {
                _textSelectionIdx = (0, m_Text.Length - 1);
            }
            _textSelectionIdxValid = true;
        }

        private void ScrollDownToSelection(Rect workingArea)
        {
            if (!workingArea.Contains(_selectDragPos + new Vector2(0, fontHeight / 2)))
            {
                GUI.ScrollTo(new Rect(_selectDragPos + new Vector2(0, fontHeight), Vector2.zero));
            }

            _textSelectionIdxValid = false;
        }

        private void ScrollUpToSelection(Rect workingArea)
        {
            if (!workingArea.Contains(_selectDragPos - new Vector2(0, fontHeight / 2)))
            {
                GUI.ScrollTo(new Rect(_selectDragPos - new Vector2(0, fontHeight), Vector2.zero));
            }

            _textSelectionIdxValid = false;
        }

        internal void MoveSelectionLeft(Rect workingArea, bool showBranchMarkers)
        {
            if (_disassembler != null)
            {
                float hPad = showBranchMarkers ? horizontalPad : naturalEnhancedPad;
                string text;
                int prevLineIdx = Mathf.FloorToInt((_selectDragPos.y - _selectEndY) / fontHeight) - 1;

                if (_selectDragPos.x <= hPad + fontWidth)
                {
                    if (prevLineIdx < 0 && _selectBlockEnd == 0)
                    {
                        // we are at the beginning of the text.
                        return;
                    }

                    if (prevLineIdx < 0 && _selectBlockEnd > 0)
                    {
                        // get text from previous block, and do calculations for that.
                        _selectBlockEnd--;

                        (text, prevLineIdx) = !_folded[_selectBlockEnd]
                            ? (_disassembler.GetOrRenderBlockToText(_selectBlockEnd), _disassembler.Blocks[_selectBlockEnd].Length - 1)
                            : (_foldedString, 0);
                    }
                    else
                    {
                        text = _disassembler.GetOrRenderBlockToText(_selectBlockEnd);
                    }
                    int charsInLine = GetEndIndexOfColoredLine(text, prevLineIdx).relative;

                    _selectDragPos.x = charsInLine * fontWidth + hPad + fontWidth / 2;
                    _selectDragPos.y -= fontHeight;
                }
                else
                {
                    // simply move selection left.
                    text = _folded[_selectBlockEnd] ? _foldedString : _disassembler.GetOrRenderBlockToText(_selectBlockEnd);
                    int charsInLine = GetEndIndexOfColoredLine(text, prevLineIdx + 1).relative;

                    if (_selectDragPos.x > charsInLine * fontWidth + hPad)
                        _selectDragPos.x = hPad + charsInLine * fontWidth + fontWidth / 2;

                    _selectDragPos.x -= fontWidth;
                }
            }
            else
            {
                int prevLineIdx = Mathf.FloorToInt((_selectDragPos.y) / fontHeight) - 1;

                if (_selectDragPos.x <= fontWidth)
                {
                    if (prevLineIdx < 0)
                    {
                        // we are at beginning of the text
                        return;
                    }

                    int charsInLine = GetEndIndexOfPlainLine(m_Text, prevLineIdx).relative;

                    _selectDragPos.x = charsInLine * fontWidth + fontWidth / 2;
                    _selectDragPos.y -= fontHeight;
                }
                else
                {
                    // simply move selection left.
                    int charsInLine = GetEndIndexOfPlainLine(m_Text, prevLineIdx+1).relative;
                    if (_selectDragPos.x > charsInLine * fontWidth)
                    {
                        _selectDragPos.x = charsInLine * fontWidth + fontWidth / 2;
                    }

                    _selectDragPos.x -= fontWidth;
                }
            }
            // check if we moved outside of view and scroll if true.
            ScrollUpToSelection(workingArea);
        }

        internal void MoveSelectionRight(Rect workingArea, bool showBranchMarkers)
        {
            if (_disassembler != null)
            {
                float hPad = showBranchMarkers ? horizontalPad : 20f;

                string text     = _disassembler.GetOrRenderBlockToText(_selectBlockEnd);
                int thisLine    = Mathf.FloorToInt((_selectDragPos.y - _selectEndY) / fontHeight);
                int charsInLine = GetEndIndexOfColoredLine(text, thisLine).relative;

                if (_selectDragPos.x >= hPad + charsInLine * fontWidth)
                {
                    // move down a line:
                    thisLine++;

                    int lineCount = _folded[_selectBlockEnd] ? 1 : _disassembler.Blocks[_selectBlockEnd].Length;

                    if (thisLine > lineCount && _selectBlockEnd == _disassembler.Blocks.Count)
                    {
                        // We are at the end of the text
                        return;
                    }

                    if (thisLine > lineCount)
                    {
                        // selected into next block.
                        _selectBlockEnd++;
                    }

                    _selectDragPos.x = hPad + fontWidth/2;
                    _selectDragPos.y += fontHeight;
                }
                else
                {
                    // simply move selection right.
                    if (_selectDragPos.x < hPad)
                    {
                        _selectDragPos.x = hPad + fontWidth / 2;
                    }

                    _selectDragPos.x += fontWidth;
                }
            }
            else
            {
                int thisLine = Mathf.FloorToInt((_selectDragPos.y) / fontHeight);

                int charsInLine = GetEndIndexOfColoredLine(m_Text, thisLine).relative;

                if (_selectDragPos.x >= charsInLine*fontWidth)
                {
                    thisLine++;

                    if (thisLine >= _mTextLines)
                    {
                        // we are at end of the text
                        return;
                    }

                    _selectDragPos.x = 0f;
                    _selectDragPos.y += fontHeight;

                }
                else
                {
                    // simply move selection right.
                    _selectDragPos.x += fontWidth;
                }
            }
            // check if we moved outside of view and scroll if true.
            ScrollDownToSelection(workingArea);
        }

        internal void MoveSelectionUp(Rect workingArea)
        {
            if (_selectDragPos.y > fontHeight)
            {
                _selectDragPos.y -= fontHeight;
            }

            // check if we moved outside of view and scroll if true.
            ScrollUpToSelection(workingArea);
        }

        internal void MoveSelectionDown(Rect workingArea)
        {
            if (_selectDragPos.y < finalAreaSize.y - fontHeight)
            {
                _selectDragPos.y += fontHeight;
            }

            // check if we moved outside of view and scroll if true.
            ScrollDownToSelection(workingArea);
        }

        internal bool MouseOutsideView(Rect workingArea, Vector2 mousePos, int controlID)
        {
            if (_mouseDown && !workingArea.Contains(mousePos))
            {
                // Mouse was dragged outside of the view.
                if (GUIUtility.hotControl == controlID && Event.current.rawType == EventType.MouseUp)
                {
                    _mouseDown = false;
                }

                _mouseOutsideBounds = true;
                return true;
            }

            _mouseOutsideBounds = false;
            return false;
        }

        internal void MouseClicked(bool withShift, Vector2 mousePos, int controlID)
        {
            if (_mouseOutsideBounds)
            {
                return;
            }

            GUIUtility.hotControl = controlID;
            // FocusControl is to take keyboard focus away from the TreeView.
            GUI.FocusControl("long text");
            if (withShift)
            {
                _selectDragPos = mousePos;
                _textSelectionIdxValid = false;
            }
            else
            {
                _selectPos = mousePos;
                StopSelection();
            }
            _mouseDown = true;
        }

        internal void DragMouse(Vector2 mousePos)
        {
            if (_mouseDown)
            {
                _selectDragPos = mousePos;
                _textSelectionIdxValid = false;
            }
        }

        internal void MouseReleased()
        {
            GUIUtility.hotControl = 0;
            _mouseDown = false;
        }

        internal void DoScroll(float mouseRelMoveY)
        {
            if (_mouseDown)
            {
                _selectDragPos.y += mouseRelMoveY * naturalEnhancedPad; // naturalEnhancedPad magic number taken from unity engine (GUI.cs under EndScrollView).
                _textSelectionIdxValid = false;
            }
        }

        private void StopSelection()
        {
            _selectDragPos = _selectPos;
            _textSelectionIdx = (0, 0);
            _textSelectionIdxValid = true;
        }

        private bool HasSelection()
        {
            return _selectPos != _selectDragPos;
        }


        /// <returns> (idx in regards to whole str, where colour tags are removed from this line, idx from this line with colour tags removed) </returns>
        internal (int total, int relative) GetEndIndexOfColoredLine(string str, int line)
        {
            int lastIdx = -1;
            int newIdx = -1;

            for (int i = 0; i <= line; i++)
            {
                lastIdx = newIdx;
                newIdx = str.IndexOf('\n', lastIdx+1);
            }
            // Remove color tag filler of line:
            lastIdx++;
            int endIdx = newIdx != -1 ? newIdx : str.Length - 1;
            int colorTagFiller = 0;
            bool lastWasStart = true;
            int colorTagStart = str.IndexOf("<color=", lastIdx);
            while (colorTagStart != -1 && colorTagStart < endIdx)
            {
                int colorTagEnd = str.IndexOf('>', colorTagStart+1);
                // +1 as the index calculation is zero based.
                colorTagFiller += colorTagEnd - colorTagStart + 1;

                if (lastWasStart)
                {
                    colorTagStart = str.IndexOf("</color>", colorTagEnd + 1);
                    lastWasStart = false;
                }
                else
                {
                    colorTagStart = str.IndexOf("<color=", colorTagEnd + 1);
                    lastWasStart = true;
                }
            }
            return (endIdx - colorTagFiller, endIdx - lastIdx - colorTagFiller);
        }

        internal (int total, int relative) GetEndIndexOfPlainLine(string str, int line)
        {
            int lastIdx = -1;
            int newIdx = -1;

            for (int i = 0; i <= line; i++)
            {
                lastIdx = newIdx;
                newIdx = str.IndexOf('\n', lastIdx+1);
            }
            lastIdx++;
            return newIdx != -1 ? (newIdx, newIdx - lastIdx) : (str.Length-1, str.Length-1 - lastIdx);
        }

        /// <summary>
        /// Checks if num is within the closed interval defined by endPoint1 and endPoint2.
        /// </summary>
        internal bool WithinRange(float endPoint1, float endPoint2, float num)
        {
            float start, end;
            if (endPoint1 < endPoint2)
            {
                start   = endPoint1;
                end     = endPoint2;
            }
            else
            {
                start   = endPoint2;
                end     = endPoint1;
            }
            return start <= num && num <= end;
        }

        /// <summary>
        /// Renders a blue box relative to text at (positionX, positionY) from start idx to end idx.
        /// </summary>
        private void RenderLineSelection(float positionX, float positionY, int start, int end)
        {
            const int alignmentPad = 2;
            var oldColor = GUI.color;
            GUI.color = _selectionColor;
            GUI.Box(new Rect(positionX + alignmentPad + start*fontWidth, positionY, (end - start)*fontWidth, fontHeight), "", textureStyle);
            GUI.color = oldColor;
        }

        /// <remarks>
        /// https://www.programmingnotes.org/7601/cs-how-to-round-a-number-to-the-nearest-x-using-cs/
        /// </remarks>
        private float RoundDownToNearest(float number, float to)
        {
            float inverse = 1 / to;
            float dividend = Mathf.Floor(number * inverse);
            return dividend / inverse;
        }

        private void SelectText(float positionX, float positionY, float fragHeight, string text, Func<string, int, (int total, int relative)> GetEndIndexOfLine)
        {
            if (HasSelection() && (WithinRange(_selectPos.y, _selectDragPos.y, positionY) || WithinRange(_selectPos.y, _selectDragPos.y, positionY + fragHeight)
                            || WithinRange(positionY, positionY + fragHeight, _selectPos.y)))
            {
                Vector2 top = _selectPos;
                Vector2 bot = _selectDragPos;
                if (_selectPos.y > _selectDragPos.y)
                {
                    top = _selectDragPos;
                    bot = _selectPos;
                }
                // fixing so we only look at things within this current fragment.
                Vector2 start = top.y < positionY ? new Vector2(positionX, positionY) : top;
                Vector2 last = bot.y < positionY + fragHeight ? bot : new Vector2(finalAreaSize.x, positionY + fragHeight - fontHeight / 2);

                int startLine = Mathf.FloorToInt((start.y - positionY) / fontHeight);
                int lastLine = Mathf.FloorToInt((last.y - positionY) / fontHeight);

                if (startLine == lastLine && start.x > last.x)
                    (start, last) = (last, start);

                // Used for making sure charsIn and charsInDrag does not exceed line length.
                var (_, startLineEndIdxRel) = GetEndIndexOfLine(text, startLine);
                var (_, lastLineEndIdxRel) = GetEndIndexOfLine(text, lastLine);

                int charsIn = Math.Min(Mathf.FloorToInt((start.x - positionX) / fontWidth), startLineEndIdxRel);
                int charsInDrag = Math.Min(Mathf.FloorToInt((last.x - positionX) / fontWidth), lastLineEndIdxRel);
                charsIn = charsIn < 0 ? 0 : charsIn;
                charsInDrag = charsInDrag < 0 ? 0 : charsInDrag;

                if (RoundDownToNearest(last.y, fontHeight) > RoundDownToNearest(start.y, fontHeight))
                {
                    // Multiline selection in this text.
                    int lineEndIdx = startLineEndIdxRel;
                    if (start.y != positionY)
                    {
                        // Selection started in this fragment.
                        RenderLineSelection(positionX, positionY + (startLine * fontHeight), charsIn, lineEndIdx + 1);
                        startLine++;
                    }

                    for (; startLine < lastLine; startLine++)
                    {
                        lineEndIdx = GetEndIndexOfLine(text, startLine).relative;
                        RenderLineSelection(positionX, positionY + (startLine * fontHeight), 0, lineEndIdx + 1);
                    }

                    if (positionY + fragHeight < bot.y)
                    {
                        // select going into next fragment
                        lineEndIdx = GetEndIndexOfLine(text, startLine).relative;
                        charsInDrag = lineEndIdx + 1;
                    }

                    RenderLineSelection(positionX, positionY + (startLine * fontHeight), 0, charsInDrag);
                }
                else
                {
                    // Single line selection in this text segment.
                    int startIdx = charsIn;
                    int endIdx = charsInDrag;
                    if (start.y == positionY)
                    {
                        // Selection started in text segment above.
                        startIdx = 0;
                    }

                    if (positionY + fragHeight < bot.y)
                    {
                        // Selection going into next text segment.
                        endIdx = startLineEndIdxRel + 1;
                    }

                    RenderLineSelection(positionX, positionY + (startLine * fontHeight), startIdx, endIdx);
                }
            }
        }

        /// <summary>
        /// Updates _textSelectionIdx based on the position of _selectPos and _selectDragPos.
        /// </summary>
        /// <param name="GetEndIndexOfLine"> either  GetEndIndexOfPlainLine or GetEndIndexOfColoredLine</param>
        private void UpdateSelectTextIdx(Func<string, int, (int total, int relative)> GetEndIndexOfLine)
        {
            if (!_textSelectionIdxValid && HasSelection())
            {
                var start = _selectPos;
                var last = _selectDragPos;
                if (last.y < start.y)
                {
                    start = _selectDragPos;
                    last = _selectPos;
                }
                int startLine   = Mathf.FloorToInt(start.y / fontHeight);
                int lastLine    = Mathf.FloorToInt(last.y / fontHeight);

                if (startLine == lastLine && start.x > last.x)
                {
                    (start, last) = (last, start);
                }

                var (startLineEndIdxTotal, startLineEndIdxRel) = GetEndIndexOfLine(m_Text, startLine);
                var (lastLineEndIdxTotal, lastLineEndIdxRel) = GetEndIndexOfLine(m_Text, lastLine);

                int charsIn = Math.Min(Mathf.FloorToInt(start.x / fontWidth), startLineEndIdxRel);
                int charsInDrag = Math.Min(Mathf.FloorToInt(last.x / fontWidth), lastLineEndIdxRel);
                charsIn = charsIn < 0 ? 0 : charsIn;
                charsInDrag = charsInDrag < 0 ? 0 : charsInDrag;

                int selectStartIdx = startLineEndIdxTotal - (startLineEndIdxRel - charsIn);
                _textSelectionIdx = (selectStartIdx, lastLineEndIdxTotal - (lastLineEndIdxRel - charsInDrag) - selectStartIdx);

                _textSelectionIdxValid = true;
            }
        }

        private void RenderPlainOldFragments(GUIStyle style, Rect workingArea)
        {
            Vector2 scrollPos = workingArea.position;
            float positionY = 0.0f;
            foreach (var fragment in m_Fragments)
            {
                float fragHeight = fragment.lineCount * fontHeight;

                if ((positionY - scrollPos.y + fragHeight) < 0)
                {
                    positionY += fragHeight;
                    continue;
                }

                if ((positionY - scrollPos.y) > workingArea.size.y)
                    break;

                // we append \n here, as we want it for the selection but not the rendering.
                SelectText(horizontalPad, positionY, fragHeight, fragment.text + '\n', GetEndIndexOfPlainLine);

                GUI.Label(new Rect(horizontalPad, positionY, finalAreaSize.x, fragHeight), fragment.text, style);
                positionY += fragHeight;
            }
            UpdateSelectTextIdx(GetEndIndexOfPlainLine);
        }

        public void Render(GUIStyle style, Rect workingArea, bool showBranchMarkers)
        {
            var hPad = showBranchMarkers ? horizontalPad : 20.0f;
            style.richText = true;

            if (invalidated)
            {
                Layout(style, hPad);
            }

            if (_disassembler != null)
            {
                // We always need to call this as its sets up the correct horizontal bar and block rendering
                LayoutEnhanced(style, workingArea, showBranchMarkers);
            }

            // Make sure finalAreaSize.x is correct prior to here
            GUILayoutUtility.GetRect(finalAreaSize.x + (showBranchMarkers?horizontalPad:20.0f),finalAreaSize.y);

            if (Event.current.type == EventType.Layout)
            {
                // working area will be valid only during repaint, for the layout event we don't draw the labels
                return;
            }

            if (_disassembler == null)
            {
                RenderPlainOldFragments(style, workingArea);
            }
            else
            {
                RenderEnhanced(style, workingArea, showBranchMarkers, hPad);
            }
            //DrawHover(style);
        }

        private void TestSelUnderscore(GUIStyle style, Vector2 scrollPos, Rect workingArea)
        {
            // TODO selection/underline items - for now still hard wired
            var current = GUI.color;

            // Selection
            GUI.color = Color.blue;
            GUI.Box(
                new Rect(horizontalPad + style.padding.left + fontWidth * 8, style.padding.top + fontHeight * 19,
                    3 * fontWidth, 1 * fontHeight), "", textureStyle);

            // Underscored
            Vector2 start = new Vector2(horizontalPad + style.padding.left + fontWidth * 8,
                style.padding.top + fontHeight * 20 - 2);
            Vector2 end = start + new Vector2((3 + 22) * fontWidth, 0 * fontHeight);

            GUI.color = Color.red;
            DrawLine(start, end, 2);

            GUI.color = current;
        }

        private void RenderBranches(Rect workingArea)
        {
            var color = GUI.color;
            List<Branch> branches = new List<Branch>();
            _hoveredBranch = default;
            for (int idx = 0;idx<_disassembler.Blocks.Count;idx++)
            {
                var block = _disassembler.Blocks[idx];
                if (block.Edges != null)
                {
                    foreach (var edge in block.Edges)
                    {
                        if (edge.Kind == BurstDisassembler.AsmEdgeKind.OutBound)
                        {
                            var srcLine = _blockLine[idx];
                            if (!_folded[idx])
                            {
                                srcLine += edge.OriginRef.LineIndex;
                            }
                            var dstBlockIdx = edge.LineRef.BlockIndex;
                            var dstLine = _blockLine[dstBlockIdx];
                            if (!_folded[dstBlockIdx])
                            {
                                dstLine += edge.LineRef.LineIndex;
                            }

                            int arrowMinY = srcLine;
                            int arrowMaxY = dstLine;
                            if (srcLine > dstLine)
                            {
                                (arrowMinY, arrowMaxY) = (dstLine, srcLine);
                            }

                            if ((dstBlockIdx == idx + 1 && edge.LineRef.LineIndex == 0) // pointing to next line
                                || !(workingArea.yMin <= arrowMaxY * fontHeight &&      // Arrow not inside view.
                                    workingArea.yMax >= arrowMinY * fontHeight))
                            {
                                continue;
                            }
                            branches.Add(CalculateBranch(edge, horizontalPad - (4 + fontWidth), srcLine * fontHeight,
                                dstLine * fontHeight, _lineDepth[idx]));
                        }
                    }
                }
            }
            foreach (var branch in branches)
            {
                if (!branch.Edge.Equals(_hoveredBranch.Edge))
                {
                    DrawBranch(branch, _lineDepth[branch.Edge.OriginRef.BlockIndex], workingArea);
                }
            }
            if (!_hoveredBranch.Edge.Equals(default(BurstDisassembler.AsmEdge)))
            {
                DrawBranch(_hoveredBranch, _lineDepth[_hoveredBranch.Edge.OriginRef.BlockIndex], workingArea);
            }

            _prevHoveredEdge = _hoveredBranch.Edge;
            GUI.color = color;
        }

        internal int BumpSelectionXByColorTag(string text, int lineIdxTotal, int charsIn)
        {
            bool lastWasStart = true;
            int colorTagStart = text.IndexOf("<color=", lineIdxTotal);

            while (colorTagStart != -1 && colorTagStart - lineIdxTotal < charsIn)
            {
                int colorTagEnd = text.IndexOf('>', colorTagStart + 1);
                // +1 as the index calculation is zero based.
                charsIn += colorTagEnd - colorTagStart + 1;

                if (lastWasStart)
                {
                    colorTagStart = text.IndexOf("</color>", colorTagEnd + 1);
                    lastWasStart = false;
                }
                else
                {
                    colorTagStart = text.IndexOf("<color=", colorTagEnd + 1);
                    lastWasStart = true;
                }
            }
            return charsIn;
        }

        private void UpdateEnhancedSelectTextIdx(float hPad)
        {
            if (!_textSelectionIdxValid && HasSelection())
            {
                int blockIdxStart = _selectBlockStart;
                int blockIdxEnd = _selectBlockEnd;
                float blockStartPosY = _selectStartY;
                float blockEndPosY = _selectEndY;

                var start = _selectPos;
                var last = _selectDragPos;
                if (last.y < start.y)
                {
                    // we selected upwards.
                    (start, last) = (last, start);
                    (blockIdxStart, blockIdxEnd, blockStartPosY, blockEndPosY) = (blockIdxEnd, blockIdxStart, blockEndPosY, blockStartPosY);
                }

                int blockStartline = Mathf.FloorToInt((start.y - blockStartPosY) / fontHeight);
                int blockEndLine = Mathf.FloorToInt((last.y - blockEndPosY) / fontHeight);

                if (blockStartline == blockEndLine && blockIdxStart == blockIdxEnd && start.x > last.x)
                {
                    // _selectDragPos was above and behind _selectPos on same line.
                    (start, last) = (last, start);
                }

                string text = _folded[blockIdxStart]
                    ? _foldedString
                    : _disassembler.GetOrRenderBlockToText(blockIdxStart);

                var (startLineEndIdxTotal, startLineEndIdxRel) = GetEndIndexOfColoredLine(text, blockStartline);
                int startOfLineIdx = startLineEndIdxTotal - startLineEndIdxRel;

                int charsIn = Math.Min(Mathf.FloorToInt((start.x - hPad) / fontWidth), startLineEndIdxRel);
                charsIn = charsIn < 0 ? 0 : charsIn;

                // Adjust charsIn so it takes color tags into considerations.
                charsIn = BumpSelectionXByColorTag(text, startOfLineIdx, charsIn + 1) - 1; // +1 -1 to not bump charsIn when selecting char just after color tag.

                _enhancedTextSelectionIdxStart = (blockIdxStart, startOfLineIdx + charsIn);


                if (blockIdxStart < blockIdxEnd)
                {
                    text = _folded[blockIdxEnd]
                        ? _foldedString
                        : _disassembler.GetOrRenderBlockToText(blockIdxEnd);
                }

                var (lastLineEndIdxTotal, lastLineEndIdxRel) = GetEndIndexOfColoredLine(text, blockEndLine);
                startOfLineIdx = lastLineEndIdxTotal - lastLineEndIdxRel;

                int charsInDrag = Math.Min(Mathf.FloorToInt((last.x - hPad) / fontWidth), lastLineEndIdxRel);
                charsInDrag = charsInDrag < 0 ? 0 : charsInDrag;

                // Adjust charsInDrag so it takes color tags into considerations.
                charsInDrag = BumpSelectionXByColorTag(text, startOfLineIdx, charsInDrag);

                _enhancedTextSelectionIdxEnd = (blockIdxEnd, startOfLineIdx + charsInDrag);

                _textSelectionIdxValid = true;
            }
        }

        private void RenderEnhanced(GUIStyle style, Rect workingArea, bool showBranchMarkers, float hPad)
        {
            //TestSelUnderscore();
            if (showBranchMarkers)
            {
                RenderBranches(workingArea);
            }

            float positionY = _renderStartY;
            for (int b = _renderBlockStart; b <= _renderBlockEnd; b++)
            {
                var block = _disassembler.Blocks[b];

                var pressed = DrawFold(hPad - 2, positionY, _folded[b], block.Kind);
                if (pressed)
                {
                    _folded[b] = !_folded[b];
                    StopSelection();
                    if (_folded[b])
                    {
                        finalAreaSize.y -= Math.Max(block.Length - 1, 1) * fontHeight;
                    }
                    else
                    {
                        finalAreaSize.y += Math.Max(block.Length - 1, 1) * fontHeight;
                    }
                }

                if (!_folded[b])
                {
                    foreach (var fragment in _blocksFragments[b])
                    {
                        var fragLineCount = fragment.lineCount;
                        var fragHeight = fragLineCount * fontHeight;

                        // we append \n here, as we want it for the selection but not the rendering.
                        if (_disassembler.IsColored)
                        {
                            SelectText(hPad, positionY, fragHeight, fragment.text + '\n', GetEndIndexOfColoredLine);
                        }
                        else
                        {
                            SelectText(hPad, positionY, fragHeight, fragment.text + '\n', GetEndIndexOfPlainLine);
                        }

                        GUI.Label(new Rect(hPad, positionY, finalAreaSize.x, fragHeight),
                            fragment.text, style);
                        positionY += fragHeight;
                    }
                }
                else
                {
                    SelectText(hPad, positionY, fontHeight, _foldedString, GetEndIndexOfPlainLine);

                    GUI.Label(new Rect(hPad, positionY, finalAreaSize.x, fontHeight), _foldedString, style);
                    positionY += fontHeight;
                }
            }
            UpdateEnhancedSelectTextIdx(hPad);
        }

        private List<Fragment> RecomputeFragments(string text)
        {
            List<Fragment> result = new List<Fragment>();

            string[] pieces = text.Split('\n');
            _mTextLines = pieces.Length;

            StringBuilder b = new StringBuilder();

            int lineCount = 0;
            for (int a=0;a<pieces.Length;a++)
            {
                if (b.Length >= kMaxFragment)
                {
                    b.Remove(b.Length - 1, 1);
                    AddFragment(b, lineCount, result);
                    lineCount = 0;
                }

                b.Append(pieces[a]);
                b.Append('\n');
                lineCount++;
            }

            if (b.Length>0)
            {
                b.Remove(b.Length - 1, 1);
                AddFragment(b, lineCount, result);
            }

            return result;
        }

        private List<Fragment> RecomputeFragmentsFromBlock(int blockIdx)
        {
            var text = _disassembler.GetOrRenderBlockToText(blockIdx).TrimEnd('\n');
            return RecomputeFragments(text);
        }

        private static void AddFragment(StringBuilder b, int lineCount, List<Fragment> result)
        {
            result.Add(new Fragment() { text = b.ToString(), lineCount = lineCount });
            b.Length = 0;
        }
    }

}