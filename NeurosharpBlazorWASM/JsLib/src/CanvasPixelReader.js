export default function parseCanvas(url, canvasRef){
    function initContext(canvasReference, contextType)
    {
        var context = canvasReference.getContext(contextType);
        return context;
    }

    function loadImage(imageSource, context)
    {
        var imageObj = new Image();
        var data;
        imageObj.onload = function()
        {
            context.drawImage(imageObj, 0, 0);
            var imageData = context.getImageData(0,0,10,10);
            readImage(imageData);
        };
        imageObj.src = imageSource;
        return imageObj;
    }

    function readImage(imageData)
    {
        DotNet.invokeMethodAsync('NeurosharpBlazorWASM', 'ReceiveImageData', imageData.data);
    }
    
    var context = initContext(canvasRef,'2d');
    var imageObj = loadImage(url, context);
}