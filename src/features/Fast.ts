/**
 * FAST intends for "Features from Accelerated Segment Test". This method
 * performs a point segment test corner detection. The segment test
 * criterion operates by considering a circle of sixteen pixels around the
 * corner candidate p. The detector classifies p as a corner if there exists
 * a set of n contiguous pixels in the circle which are all brighter than the
 * intensity of the candidate pixel Ip plus a threshold t, or all darker
 * than Ip − t.
 *
 *       15 00 01
 *    14          02
 * 13                03
 * 12       []       04
 * 11                05
 *    10          06
 *       09 08 07
 *
 * For more reference:
 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.60.3991&rep=rep1&type=pdf
 */
export class Fast {
    /**
     * Holds the threshold to determine whether the tested pixel is brighter or
     * darker than the corner candidate p.
     */
    public static readonly THRESHOLD: number = 40;

    /**
     * Caches coordinates values of the circle surrounding the pixel candidate p.
     */
    private static circles_: Record<number, Int32Array> = {};

    /**
     * Finds corners coordinates on the grayscaled image.
     * @param pixels The grayscale pixels in a linear [p1,p2,...] array.
     * @param width The image width.
     * @param height The image height.
     * @param threshold to determine whether the tested pixel is brighter or
     *     darker than the corner candidate p. Default value is 40.
     * @return Array containing the coordinates of all found corners,
     *     e.g. [x0,y0,x1,y1,...], where P(x0,y0) represents a corner coordinate.
     */
    public static findCorners(pixels: number[], width: number, height: number, threshold?: number): number[] {
        const circleOffsets = this.getCircleOffsets_(width);
        const circlePixels = new Int32Array(16);
        const corners: number[] = [];

        const optThreshold = threshold !== undefined ? threshold : this.THRESHOLD;

        // When looping through the image pixels, skips the first three lines from
        // the image boundaries to constrain the surrounding circle inside the image
        // area.
        for (let i = 3; i < height - 3; i++) {
            for (let j = 3; j < width - 3; j++) {
                const w = i * width + j;
                const p = pixels[w];

                // Loops the circle offsets to read the pixel value for the sixteen
                // surrounding pixels.
                for (let k = 0; k < 16; k++) {
                    circlePixels[k] = pixels[w + circleOffsets[k]];
                }

                if (this.isCorner(p, circlePixels, optThreshold)) {
                    // The pixel p is classified as a corner, as optimization increment j
                    // by the circle radius 3 to skip the neighbor pixels inside the
                    // surrounding circle. This can be removed without compromising the
                    // result.
                    corners.push(j, i);
                    j += 3;
                }
            }
        }

        return corners;
    }

    /**
     * Checks if the circle pixel is brighter than the candidate pixel p by
     * a threshold.
     * @param circlePixel The circle pixel value.
     * @param p The value of the candidate pixel p.
     * @param threshold
     * @return Boolean
     */
    public static isBrighter(circlePixel: number, p: number, threshold: number): boolean {
        return circlePixel - p > threshold;
    }

    /**
     * Checks if the circle pixel is within the corner of the candidate pixel p
     * by a threshold.
     * @param p The value of the candidate pixel p.
     * @param circlePixels The circle pixel values.
     * @param threshold
     * @return Boolean
     */
    public static isCorner(p: number, circlePixels: Int32Array, threshold: number): boolean {
        if (this.isTriviallyExcluded(circlePixels, p, threshold)) {
            return false;
        }

        for (let x = 0; x < 16; x++) {
            let darker = true;
            let brighter = true;

            for (let y = 0; y < 9; y++) {
                const circlePixel = circlePixels[(x + y) & 15];

                if (!this.isBrighter(p, circlePixel, threshold)) {
                    brighter = false;
                    if (darker === false) {
                        break;
                    }
                }

                if (!this.isDarker(p, circlePixel, threshold)) {
                    darker = false;
                    if (brighter === false) {
                        break;
                    }
                }
            }

            if (brighter || darker) {
                return true;
            }
        }

        return false;
    }

    /**
     * Checks if the circle pixel is darker than the candidate pixel p by
     * a threshold.
     * @param circlePixel The circle pixel value.
     * @param p The value of the candidate pixel p.
     * @param threshold
     * @return Boolean
     */
    public static isDarker(circlePixel: number, p: number, threshold: number): boolean {
        return p - circlePixel > threshold;
    }

    /**
     * Fast check to test if the candidate pixel is a trivially excluded value.
     * In order to be a corner, the candidate pixel value should be darker or
     * brighter than 9-12 surrounding pixels, when at least three of the top,
     * bottom, left and right pixels are brighter or darker it can be
     * automatically excluded improving the performance.
     * @param circlePixels The circle pixel values.
     * @param p The value of the candidate pixel p.
     * @param threshold
     * @return Boolean
     */
    public static isTriviallyExcluded(circlePixels: Int32Array, p: number, threshold: number): boolean {
        let count = 0;
        const circleBottom = circlePixels[8];
        const circleLeft = circlePixels[12];
        const circleRight = circlePixels[4];
        const circleTop = circlePixels[0];

        if (this.isBrighter(circleTop, p, threshold)) {
            count++;
        }
        if (this.isBrighter(circleRight, p, threshold)) {
            count++;
        }
        if (this.isBrighter(circleBottom, p, threshold)) {
            count++;
        }
        if (this.isBrighter(circleLeft, p, threshold)) {
            count++;
        }

        if (count < 3) {
            count = 0;
            if (this.isDarker(circleTop, p, threshold)) {
                count++;
            }
            if (this.isDarker(circleRight, p, threshold)) {
                count++;
            }
            if (this.isDarker(circleBottom, p, threshold)) {
                count++;
            }
            if (this.isDarker(circleLeft, p, threshold)) {
                count++;
            }
            if (count < 3) {
                return true;
            }
        }

        return false;
    }

    /**
     * Gets the sixteen offset values of the circle surrounding pixel.
     * @param width The image width.
     * @return Array with the sixteen offset values of the circle
     *     surrounding pixel.
     */
    private static getCircleOffsets_(width: number): Int32Array {
        if (this.circles_[width]) {
            return this.circles_[width];
        }

        const circle = new Int32Array(16);

        circle[0] = -width - width - width;
        circle[1] = circle[0] + 1;
        circle[2] = circle[1] + width + 1;
        circle[3] = circle[2] + width + 1;
        circle[4] = circle[3] + width;
        circle[5] = circle[4] + width;
        circle[6] = circle[5] + width - 1;
        circle[7] = circle[6] + width - 1;
        circle[8] = circle[7] - 1;
        circle[9] = circle[8] - 1;
        circle[10] = circle[9] - width - 1;
        circle[11] = circle[10] - width - 1;
        circle[12] = circle[11] - width;
        circle[13] = circle[12] - width;
        circle[14] = circle[13] - width + 1;
        circle[15] = circle[14] - width + 1;

        this.circles_[width] = circle;
        return circle;
    }
}
