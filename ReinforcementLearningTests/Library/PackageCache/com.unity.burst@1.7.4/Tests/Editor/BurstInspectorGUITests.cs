using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Unity.Burst.Editor;
using UnityEditor;
using UnityEngine;
using UnityEngine.TestTools;
using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;

[TestFixture]
[UnityPlatform(RuntimePlatform.WindowsEditor, RuntimePlatform.OSXEditor)]
public class BurstInspectorGUITests
{
    private readonly WaitUntil _waitForInitialized =
        new WaitUntil(() => EditorWindow.GetWindow<BurstInspectorGUI>()._initialized);

    [UnitySetUp]
    public IEnumerator SetUp()
    {
        EditorWindow.GetWindow<BurstInspectorGUI>().Show();

        // Make sure window is actually initialized before continuing.
        yield return _waitForInitialized;
    }

    [UnityTest]
    public IEnumerator TestInspectorOpenDuringDomainReloadDoesNotLogErrors()
    {
        // Show Inspector window
        EditorWindow.GetWindow<BurstInspectorGUI>().Show();

        Assert.IsTrue(EditorWindow.HasOpenInstances<BurstInspectorGUI>());

        // Ask for domain reload
        EditorUtility.RequestScriptReload();

        // Wait for the domain reload to be completed
        yield return new WaitForDomainReload();

        Assert.IsTrue(EditorWindow.HasOpenInstances<BurstInspectorGUI>());

        // Hide Inspector window
        EditorWindow.GetWindow<BurstInspectorGUI>().Close();

        Assert.IsFalse(EditorWindow.HasOpenInstances<BurstInspectorGUI>());
    }

    [UnityTest]
    public IEnumerator DisassemblerNotChangingUnexpectedlyTest()
    {
        var window = EditorWindow.GetWindow<BurstInspectorGUI>();

        // Selecting a specific assembly.
        window._treeView.TrySelectByDisplayName("BurstInspectorGUITests.MyJob - (IJob)");

        // Sending event to set the displayname, to avoid it resetting _scrollPos because of target change.
        window.SendEvent(new Event() { type = EventType.Repaint, mousePosition = new Vector2(window.position.width / 2f, window.position.height / 2f) });
        yield return null;

        // Doing actual test work:
        var prev = new BurstDisassemblerWithCopy(window._burstDisassembler);
        window.SendEvent(new Event() { type = EventType.Repaint, mousePosition = new Vector2(window.position.width / 2f, window.position.height / 2f) });
        yield return null;
        Assert.IsTrue(prev.Equals(window._burstDisassembler), "Public fields changed in burstDisassembler even though they shouldn't");

        prev = new BurstDisassemblerWithCopy(window._burstDisassembler);
        window.SendEvent(new Event() { type = EventType.MouseUp, mousePosition = Vector2.zero });
        yield return null;
        Assert.IsTrue(prev.Equals(window._burstDisassembler), "Public fields changed in burstDisassembler even though they shouldn't");

        prev = new BurstDisassemblerWithCopy(window._burstDisassembler);
        window._treeView.TrySelectByDisplayName("BurstReflectionTests.MyJob - (IJob)");  // Changing job, meaning SetText(.) should be called during next event.
        window.SendEvent(new Event() { type = EventType.Repaint, mousePosition = new Vector2(window.position.width / 2f, window.position.height / 2f) });
        yield return null;
        Assert.IsFalse(prev.Equals(window._burstDisassembler), "Public fields of burstDisassembler did not change");

        window.Close();
    }

    [UnityTest]
    public IEnumerator InspectorStallingLoadTest()
    {
        var win = EditorWindow.GetWindow<BurstInspectorGUI>();

        // Error was triggered by selecting a display name, filtering it out, and then doing a script recompilation.
        win._treeView.TrySelectByDisplayName("BurstInspectorGUITests.MyJob - (IJob)");
        win._searchFieldJobs.SetFocus();
        yield return null;

        // Simulate event for sending "a" as it will filter out the chosen job.
        win.SendEvent(Event.KeyboardEvent("a"));
        yield return null;

        // Send RequestScriptReload to try and trigger the bug
        // and wait for it to return
        EditorUtility.RequestScriptReload();
        yield return new WaitForDomainReload();

        win = EditorWindow.GetWindow<BurstInspectorGUI>();
        // Wait for it to actually initialize.
        yield return _waitForInitialized;

        Assert.IsTrue(win._initialized, "BurstInspector did not initialize properly after script reload");

        win.Close();
    }

    [UnityTest]
    public IEnumerator FontStyleDuringDomainReloadTest()
    {
        // Enter play mod
        yield return new EnterPlayMode();

        // Exit play mode
        yield return new ExitPlayMode();

        // Wait for the inspector to actually reload
        yield return _waitForInitialized;

        if (Application.platform == RuntimePlatform.WindowsEditor)
        {
            Assert.AreEqual("Consolas", EditorWindow.GetWindow<BurstInspectorGUI>()._font.name);
        }
        else
        {
            Assert.AreEqual("Courier", EditorWindow.GetWindow<BurstInspectorGUI>()._font.name);
        }
        EditorWindow.GetWindow<BurstInspectorGUI>().Close();
    }


    [BurstCompile]
    private struct MyJob : IJob
    {
        [ReadOnly]
        public NativeArray<float> Inpút;

        [WriteOnly]
        public NativeArray<float> Output;

        public void Execute()
        {
            float result = 0.0f;
            for (int i = 0; i < Inpút.Length; i++)
            {
                result += Inpút[i];
            }
            Output[0] = result;
        }
    }

    private class BurstDisassemblerWithCopy : BurstDisassembler
    {
        public List<AsmBlock> BlocksCopy;
        public bool IsColoredCopy;
        public List<AsmLine> LinesCopy;
        public List<AsmToken> TokensCopy;
        public BurstDisassemblerWithCopy(BurstDisassembler disassembler) : base()
        {
            IsColoredCopy = disassembler.IsColored;

            BlocksCopy = new List<AsmBlock>(disassembler.Blocks);
            LinesCopy = new List<AsmLine>(disassembler.Lines);
            TokensCopy = new List<AsmToken>(disassembler.Tokens);
        }

        public bool Equals(BurstDisassembler other)
        {
            return IsColoredCopy == other.IsColored
                && BlocksCopy.SequenceEqual(other.Blocks)
                && LinesCopy.SequenceEqual(other.Lines)
                && TokensCopy.SequenceEqual(other.Tokens);
        }
    }
}
