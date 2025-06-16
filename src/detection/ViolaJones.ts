import { Image } from "../utils/Image";
import { eye } from "./training/haar/eye";
import { mouth } from "./training/haar/mouth";
import { face } from "./training/haar/face";
import { DisjointSet } from "../utils/DisjointSet";
import { TrackingMath } from "../math/TrackingMath";
import { Rect } from "./Rect";

/**
 * ViolaJones utility class for object detection using HAAR cascade classifiers.
 */
export class ViolaJones {
    /**
     * Holds the minimum area of intersection that defines when a rectangle is
     * from the same group. Often when a face is matched multiple rectangles are
     * classified as possible rectangles to represent the face, when they
     * intersects they are grouped as one face.
     */
    public static readonly REGIONS_OVERLAP: number = 0.5;

    /**
     * Holds the HAAR cascade classifiers converted from OpenCV training.
     */
    public static classifiers: Record<string, Float64Array> = {
        eye: eye,
        mouth: mouth,
        face: face,
    };

    /**
     * Detects through the HAAR cascade data rectangles matches.
     * @param pixels The pixels in a linear [r,g,b,a,...] array.
     * @param width The image width.
     * @param height The image height.
     * @param initialScale The initial scale to start the block scaling.
     * @param scaleFactor The scale factor to scale the feature block.
     * @param stepSize The block step size.
     * @param edgesDensity Percentage density edges inside the classifier block.
     *     Value from [0.0, 1.0], defaults to 0.2. If specified edge detection
     *     will be applied to the image to prune dead areas of the image, this
     *     can improve significantly performance.
     * @param data The HAAR cascade data.
     * @returns Found rectangles.
     */
    public static detect(
        pixels: Uint8ClampedArray,
        width: number,
        height: number,
        initialScale: number,
        scaleFactor: number,
        stepSize: number,
        edgesDensity: number,
        data: Float64Array
    ): Rect[] {
        let total = 0;
        const rects: Rect[] = [];
        const integralImage = new Int32Array(width * height);
        const integralImageSquare = new Int32Array(width * height);
        const tiltedIntegralImage = new Int32Array(width * height);

        let integralImageSobel: Int32Array | undefined;
        if (edgesDensity > 0) {
            integralImageSobel = new Int32Array(width * height);
        }

        Image.computeIntegralImage(
            pixels,
            width,
            height,
            integralImage,
            integralImageSquare,
            tiltedIntegralImage,
            integralImageSobel
        );

        const minWidth = data[0];
        const minHeight = data[1];
        let scale = initialScale * scaleFactor;
        let blockWidth = (scale * minWidth) | 0;
        let blockHeight = (scale * minHeight) | 0;

        while (blockWidth < width && blockHeight < height) {
            const step = (scale * stepSize + 0.5) | 0;
            for (let i = 0; i < height - blockHeight; i += step) {
                for (let j = 0; j < width - blockWidth; j += step) {
                    if (edgesDensity > 0 && integralImageSobel) {
                        if (
                            this.isTriviallyExcluded(
                                edgesDensity,
                                integralImageSobel,
                                i,
                                j,
                                width,
                                blockWidth,
                                blockHeight
                            )
                        ) {
                            continue;
                        }
                    }

                    if (
                        this.evalStages_(
                            data,
                            integralImage,
                            integralImageSquare,
                            tiltedIntegralImage,
                            i,
                            j,
                            width,
                            blockWidth,
                            blockHeight,
                            scale
                        )
                    ) {
                        rects[total++] = {
                            width: blockWidth,
                            height: blockHeight,
                            x: j,
                            y: i,
                            total: 0,
                        };
                    }
                }
            }

            scale *= scaleFactor;
            blockWidth = (scale * minWidth) | 0;
            blockHeight = (scale * minHeight) | 0;
        }
        return this.mergeRectangles_(rects);
    }

    /**
     * Fast check to test whether the edges density inside the block is greater
     * than a threshold, if true it tests the stages. This can improve
     * significantly performance.
     * @param edgesDensity Percentage density edges inside the classifier block.
     * @param integralImageSobel The integral image of a sobel image.
     * @param i Vertical position of the pixel to be evaluated.
     * @param j Horizontal position of the pixel to be evaluated.
     * @param width The image width.
     * @param blockWidth The block width.
     * @param blockHeight The block height.
     * @returns True whether the block at position i,j can be skipped, false otherwise.
     */
    public static isTriviallyExcluded(
        edgesDensity: number,
        integralImageSobel: Int32Array,
        i: number,
        j: number,
        width: number,
        blockWidth: number,
        blockHeight: number
    ): boolean {
        const wbA = i * width + j;
        const wbB = wbA + blockWidth;
        const wbD = wbA + blockHeight * width;
        const wbC = wbD + blockWidth;
        const blockEdgesDensity =
            (integralImageSobel[wbA] - integralImageSobel[wbB] - integralImageSobel[wbD] + integralImageSobel[wbC]) /
            (blockWidth * blockHeight * 255);
        if (blockEdgesDensity < edgesDensity) {
            return true;
        }
        return false;
    }

    /**
     * Evaluates if the block size on i,j position is a valid HAAR cascade stage.
     * @param data The HAAR cascade data.
     * @param integralImage The integral image.
     * @param integralImageSquare The integral image square.
     * @param tiltedIntegralImage The tilted integral image.
     * @param i Vertical position of the pixel to be evaluated.
     * @param j Horizontal position of the pixel to be evaluated.
     * @param width The image width.
     * @param blockWidth The block width.
     * @param blockHeight The block height.
     * @param scale The scale factor of the block size and its original size.
     * @returns Whether the region passes all the stage tests.
     */
    private static evalStages_(
        data: Float64Array,
        integralImage: Int32Array,
        integralImageSquare: Int32Array,
        tiltedIntegralImage: Int32Array,
        i: number,
        j: number,
        width: number,
        blockWidth: number,
        blockHeight: number,
        scale: number
    ): boolean {
        const inverseArea = 1.0 / (blockWidth * blockHeight);
        const wbA = i * width + j;
        const wbB = wbA + blockWidth;
        const wbD = wbA + blockHeight * width;
        const wbC = wbD + blockWidth;
        const mean = (integralImage[wbA] - integralImage[wbB] - integralImage[wbD] + integralImage[wbC]) * inverseArea;
        const variance =
            (integralImageSquare[wbA] -
                integralImageSquare[wbB] -
                integralImageSquare[wbD] +
                integralImageSquare[wbC]) *
                inverseArea -
            mean * mean;

        let standardDeviation = 1;
        if (variance > 0) {
            standardDeviation = Math.sqrt(variance);
        }

        const length = data.length;

        for (let w = 2; w < length; ) {
            let stageSum = 0;
            const stageThreshold = data[w++];
            let nodeLength = data[w++];

            while (nodeLength--) {
                let rectsSum = 0;
                const tilted = data[w++];
                const rectsLength = data[w++];

                for (let r = 0; r < rectsLength; r++) {
                    const rectLeft = (j + data[w++] * scale + 0.5) | 0;
                    const rectTop = (i + data[w++] * scale + 0.5) | 0;
                    const rectWidth = (data[w++] * scale + 0.5) | 0;
                    const rectHeight = (data[w++] * scale + 0.5) | 0;
                    const rectWeight = data[w++];

                    let w1: number;
                    let w2: number;
                    let w3: number;
                    let w4: number;
                    if (tilted) {
                        // RectSum(r) = RSAT(x-h+w, y+w+h-1) + RSAT(x, y-1) - RSAT(x-h, y+h-1) - RSAT(x+w, y+w-1)
                        w1 = rectLeft - rectHeight + rectWidth + (rectTop + rectWidth + rectHeight - 1) * width;
                        w2 = rectLeft + (rectTop - 1) * width;
                        w3 = rectLeft - rectHeight + (rectTop + rectHeight - 1) * width;
                        w4 = rectLeft + rectWidth + (rectTop + rectWidth - 1) * width;
                        rectsSum +=
                            (tiltedIntegralImage[w1] +
                                tiltedIntegralImage[w2] -
                                tiltedIntegralImage[w3] -
                                tiltedIntegralImage[w4]) *
                            rectWeight;
                    } else {
                        // RectSum(r) = SAT(x-1, y-1) + SAT(x+w-1, y+h-1) - SAT(x-1, y+h-1) - SAT(x+w-1, y-1)
                        w1 = rectTop * width + rectLeft;
                        w2 = w1 + rectWidth;
                        w3 = w1 + rectHeight * width;
                        w4 = w3 + rectWidth;
                        rectsSum +=
                            (integralImage[w1] - integralImage[w2] - integralImage[w3] + integralImage[w4]) *
                            rectWeight;
                    }
                }

                const nodeThreshold = data[w++];
                const nodeLeft = data[w++];
                const nodeRight = data[w++];

                if (rectsSum * inverseArea < nodeThreshold * standardDeviation) {
                    stageSum += nodeLeft;
                } else {
                    stageSum += nodeRight;
                }
            }

            if (stageSum < stageThreshold) {
                return false;
            }
        }
        return true;
    }

    /**
     * Postprocess the detected sub-windows in order to combine overlapping
     * detections into a single detection.
     * @param rects Array of rectangles to merge.
     * @returns Merged rectangles.
     */
    private static mergeRectangles_(rects: Rect[]): Rect[] {
        // Note: This assumes tracking.DisjointSet and tracking.Math exist
        // You may need to import or implement these separately
        const disjointSet = new DisjointSet(rects.length);

        for (let i = 0; i < rects.length; i++) {
            const r1 = rects[i];
            for (let j = 0; j < rects.length; j++) {
                const r2 = rects[j];
                if (
                    TrackingMath.intersectRect(
                        r1.x,
                        r1.y,
                        r1.x + r1.width,
                        r1.y + r1.height,
                        r2.x,
                        r2.y,
                        r2.x + r2.width,
                        r2.y + r2.height
                    )
                ) {
                    const x1 = Math.max(r1.x, r2.x);
                    const y1 = Math.max(r1.y, r2.y);
                    const x2 = Math.min(r1.x + r1.width, r2.x + r2.width);
                    const y2 = Math.min(r1.y + r1.height, r2.y + r2.height);
                    const overlap = (x1 - x2) * (y1 - y2);
                    const area1 = r1.width * r1.height;
                    const area2 = r2.width * r2.height;

                    if (
                        overlap / (area1 * (area1 / area2)) >= this.REGIONS_OVERLAP &&
                        overlap / (area2 * (area1 / area2)) >= this.REGIONS_OVERLAP
                    ) {
                        disjointSet.union(i, j);
                    }
                }
            }
        }

        const map: Record<number, Rect> = {};
        for (let k = 0; k < disjointSet.length; k++) {
            const rep = disjointSet.find(k);
            if (!map[rep]) {
                map[rep] = {
                    total: 1,
                    width: rects[k].width,
                    height: rects[k].height,
                    x: rects[k].x,
                    y: rects[k].y,
                };
                continue;
            }
            map[rep].total++;
            map[rep].width += rects[k].width;
            map[rep].height += rects[k].height;
            map[rep].x += rects[k].x;
            map[rep].y += rects[k].y;
        }

        const result: Array<{
            total: number;
            width: number;
            height: number;
            x: number;
            y: number;
        }> = [];
        Object.keys(map).forEach(function (key) {
            const rect = map[parseInt(key)];
            result.push({
                total: rect.total,
                width: (rect.width / rect.total + 0.5) | 0,
                height: (rect.height / rect.total + 0.5) | 0,
                x: (rect.x / rect.total + 0.5) | 0,
                y: (rect.y / rect.total + 0.5) | 0,
            });
        });

        return result;
    }
}
