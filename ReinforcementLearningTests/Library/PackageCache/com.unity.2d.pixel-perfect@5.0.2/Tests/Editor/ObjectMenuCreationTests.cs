using NUnit.Framework;
using UnityEngine.TestTools;
using UnityEditor;

namespace UnityEngine.U2D
{
    internal class ObjectMenuCreationTests
    {
        [Test]
        public void ExecuteMenuCommandCreatesGameObjectWithPixelPerfectCamera()
        {
            var transformCount = Object.FindObjectsOfType<Transform>();
#if ENABLE_URP_14_0_0_OR_NEWER
            EditorApplication.ExecuteMenuItem("GameObject/2D Object/Pixel Perfect Camera (URP)");
#else
            EditorApplication.ExecuteMenuItem("GameObject/2D Object/Pixel Perfect Camera");
#endif
            LogAssert.NoUnexpectedReceived();
            Assert.True(Object.FindObjectsOfType<Transform>().Length > transformCount.Length);
        }
    }
}
