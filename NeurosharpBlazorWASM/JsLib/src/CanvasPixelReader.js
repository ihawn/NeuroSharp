export default function parseCanvas(url, canvasRef) {
    function initContext(canvasReference, contextType) {
        var context = canvasReference.getContext(contextType);
        return context;
    }

    function loadImage(imageSource, context) {
        var imageObj = new Image();
        var data;
        imageObj.onload = function() {
            var width = context.canvas.getAttribute("width");
            var height = context.canvas.getAttribute("height");
            context.clearRect(0, 0, width, height);
            context.drawImage(imageObj, 0, 0);
            var imageData = context.getImageData(0,0, width, height);
            readImage(imageData);
        };
        imageObj.src = imageSource;
        return imageObj;
    }

    function readImage(imageData) {
        var grayscaleData = [];
        var n = 0;
        for(var i = 3; i < imageData.data.length; i+=4){
            grayscaleData[n] = imageData.data[i];
            n++;
        }
        DotNet.invokeMethodAsync('NeurosharpBlazorWASM', 'ReceiveImageData', grayscaleData);
    }
    
    var context = initContext(canvasRef,'2d');
    var imageObj = loadImage(url, context);
}