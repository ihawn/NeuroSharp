using NUnit.Framework;
using UnityEngine;
using Unity.Burst.Editor;

[TestFixture]
public class LongTextAreaTests
{
    private LongTextArea _textArea;

    [OneTimeSetUp]
    public void SetUp()
    {
        _textArea = new LongTextArea();
    }

    [Test]
    [TestCase("", "        push        rbp\n        .seh_pushreg rbp\n", 7, true)]
    [TestCase("<color=#CCCCCC>", "        push        rbp\n        .seh_pushreg rbp\n", 25, true)]
    [TestCase("<color=#d7ba7d>", "        push        rbp\n        .seh_pushreg rbp\n", 21+15+8+15, true)]
    [TestCase("", "\n# hulahop    hejsa\n", 5, false)]
    public void GetStartingColorTagTest(string tag, string text, int textIdx, bool syntaxHighlight)
    {
        var disAssembler = new BurstDisassembler();
        _textArea.SetText(text, true, false, disAssembler, disAssembler.Initialize(text, BurstDisassembler.AsmKind.Intel, true, syntaxHighlight));

        // This should be set according to the colour tagged string.
        _textArea._enhancedTextSelectionIdxStart = (0, textIdx);

        Assert.That(_textArea.GetStartingColorTag(disAssembler.GetOrRenderBlockToText(0)), Is.EqualTo(tag));
    }

    [Test]
    [TestCase("", "        push        rbp\n        .seh_pushreg rbp\n", 7, true)]
    [TestCase("</color>", "        push        rbp\n        .seh_pushreg rbp\n", 25, true)]
    [TestCase("</color>", "        push        rbp\n        .seh_pushreg rbp\n", 21 + 15 + 8 + 15, true)]
    [TestCase("", "        push        rbp\n        .seh_pushreg rbp\n", 14 + 15 + 8, true)]
    [TestCase("", "\n# hulahop    hejsa\n", 5, false)]
    public void GetEndingColorTagTest(string tag, string text, int textIdx, bool syntaxHighlight)
    {
        var disAssembler = new BurstDisassembler();
        _textArea.SetText(text, true, false, disAssembler, disAssembler.Initialize(text, BurstDisassembler.AsmKind.Intel, true, syntaxHighlight));

        // This should be set according to the colour tagged string.
        _textArea._enhancedTextSelectionIdxEnd = (0, textIdx);

        Assert.That(_textArea.GetEndingColorTag(disAssembler.GetOrRenderBlockToText(0)), Is.EqualTo(tag));
    }

    [Test]
    [TestCase("<color=#FFFF00>hulahop</color>    <color=#DCDCAA>hejsa</color>\n", 0, 16, 16)]
    [TestCase("<color=#FFFF00>hulahop</color>\n    <color=#DCDCAA>hejsa</color>\n", 1, 40, 9)]
    [TestCase("<color=#FFFF00>hulahop</color>\n    <color=#DCDCAA>hejsa</color>\n hej", 2, 67, 3)]
    [TestCase("<color=#FFFF00>hulahop</color>    <color=#DCDCAA>hejsa</color>", 0, 15, 15)]
    [TestCase("\n        <color=#4EC9B0>je</color>                <color=#d4d4d4>.LBB11_4</color>", 1, 34, 33)]
    // Test cases for when on enhanced text and not coloured.
    [TestCase("hulahop    hejsa\n", 0, 16, 16)]
    [TestCase("hulahop\n    hejsa\n", 1, 17, 9)]
    [TestCase("hulahop\n    hejsa\n hej", 2, 21, 3)]
    [TestCase("hulahop    hejsa", 0, 15, 15)]
    public void GetEndIndexOfColoredLineTest(string text, int line, int resTotal, int resRel)
    {
        Assert.That(_textArea.GetEndIndexOfColoredLine(text, line), Is.EqualTo((resTotal, resRel)));
    }

    [Test]
    [TestCase("hulahop    hejsa\n", 0, 16, 16)]
    [TestCase("hulahop\n    hejsa\n", 1, 17, 9)]
    [TestCase("hulahop\n    hejsa\n hej", 2, 21, 3)]
    [TestCase("hulahop    hejsa", 0, 15, 15)]
    [TestCase("\nhulahop    hejsa", 1, 16, 15)]
    public void GetEndIndexOfPlainLineTest(string text, int line, int resTotal, int resRel)
    {
        Assert.That(_textArea.GetEndIndexOfPlainLine(text, line), Is.EqualTo((resTotal, resRel)));
    }

    [Test]
    [TestCase(1f, 3f, 3f, true)]
    [TestCase(1f, 3f, 2f, true)]
    [TestCase(1f, 3f, 3.00001f, false)]
    public void WithinRangeTest(float start, float end, float num, bool res)
    {
        Assert.That(_textArea.WithinRange(start, end, num), Is.EqualTo(res));
    }

    [Test]
    [TestCase("<color=#FFFF00>hulahop</color>\n    <color=#DCDCAA>hejsa</color>\n hej", 2, 2, 0)]
    [TestCase("<color=#FFFF00>hulahop</color>\n    <color=#DCDCAA>hejsa</color>\n hej", 1, 5, 15)]
    [TestCase("<color=#FFFF00>hulahop</color>    <color=#DCDCAA>hejsa</color>:", 0, 17, 46)]
    public void BumpSelectionXByColortagTest(string text, int lineNum, int charsIn, int colourTagFiller)
    {
        var (idxTotal, idxRel) = _textArea.GetEndIndexOfColoredLine(text, lineNum);
        Assert.That(_textArea.BumpSelectionXByColorTag(text, idxTotal - idxRel, charsIn), Is.EqualTo(charsIn + colourTagFiller));
    }

    [Test]
    [TestCase("        push        rbp\n        .seh_pushreg rbp\n", false)]
    [TestCase("        push        rbp\n        .seh_pushreg rbp\n", true)]
    public void SelectAllTest(string text, bool useDisassembler)
    {
        if (useDisassembler)
        {
            var disAssembler = new BurstDisassembler();
            _textArea.SetText(text, true, false, disAssembler, disAssembler.Initialize(text, BurstDisassembler.AsmKind.Intel));
        } else
        {
            _textArea.SetText(text, true, true, null, false);
        }


        _textArea._selectPos = new Vector2(2, 2);
        // There is no inserted comments or similar in my test example, so finalAreaSize, should be equivalent for the two.
        _textArea.finalAreaSize = new Vector2(7.5f * text.Length, 15.2f);

        _textArea.SelectAll();
        Assert.That(_textArea._selectPos, Is.EqualTo(Vector2.zero));
        Assert.That(_textArea._selectDragPos, Is.EqualTo(new Vector2(7.5f * text.Length, 15.2f)));

        if (!useDisassembler)
        {
            Assert.That(_textArea._textSelectionIdx, Is.EqualTo((0, text.Length - 1)));
        }
    }
}



