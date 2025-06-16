/**
 * Canvas utility class for image loading operations.
 */
export class Canvas {
    /**
     * Loads an image source into the canvas.
     * @param canvas The canvas dom element.
     * @param src The image source.
     * @param x The canvas horizontal coordinate to load the image.
     * @param y The canvas vertical coordinate to load the image.
     * @param width The image width.
     * @param height The image height.
     * @param optCallback Optional callback that fires when the image is loaded into the canvas.
     */
    static loadImage(
        canvas: HTMLCanvasElement,
        src: string,
        x: number,
        y: number,
        width: number,
        height: number,
        optCallback?: () => void
    ): void {
        const img = new Image();
        img.crossOrigin = "anonymous";

        img.onload = function () {
            const context = canvas.getContext("2d");
            if (context) {
                canvas.width = width;
                canvas.height = height;
                context.drawImage(img, x, y, width, height);

                if (optCallback) {
                    optCallback();
                }
            }
        };

        img.src = src;
    }
}
