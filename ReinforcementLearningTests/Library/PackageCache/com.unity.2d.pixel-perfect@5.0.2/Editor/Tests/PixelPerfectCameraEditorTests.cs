#if ENABLE_URP_14_0_0_OR_NEWER
using NUnit.Framework;
using System.Collections.Generic;
using System.Linq;
using PackageManager = UnityEditor.PackageManager;

namespace UnityEngine.U2D
{
    internal class PixelPerfectCameraTests
    {
        [Test]
        public void UpgradeToLatestURPVersion()
        {
            GameObject go = new GameObject();
            var cam = go.AddComponent<PixelPerfectCamera>();

            // Check if upgrade is successful
            Assert.True(UnityEditor.Rendering.Universal.U2DToURPPixelPerfectConverter.UpgradePixelPerfectCamera(cam), "Failed to upgrade to URP Pixel Perfect Camera.");

            // Make sure game object no longer has 2d package pixel perfect camera
            var res = go.GetComponent<PixelPerfectCamera>() == null;
            Assert.True(res, "Game object should not have a Pixel Perfect Camera from the 2D Package after upgrading.");
        }
    }
}
#endif
